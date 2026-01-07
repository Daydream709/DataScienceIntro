import os
import time
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 配置参数 (更新为 Top 3 专用路径)
# ==========================================
class Config:
    # 调优后的四个 Top 3 模型预测结果路径
    MODEL_PREDS = {
        "CatBoost": "../../result/cat_top3_tuning_result/cat_tuned_preds_top3.csv",
        "LightGBM": "../../result/lgbm_top3_tuning_result/lgbm_tuned_preds_top3.csv",
        "XGBoost": "../../result/xgb_top3_tuning_result/xgb_tuned_preds_top3.csv",
        "TabNet": "../../result/tabnet_top3_result/tabnet_preds_top3.csv",
    }

    # Top 3 标签路径 (提取最后一列)
    LABEL_PATH = "../../data/y_test_final_top3.csv"

    # 输出目录
    OUTPUT_DIR = "../../result/blending_top3_tuning_result"
    N_TRIALS = 150  # 增加迭代次数以获得更精细的比例


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 载入数据
# ==========================================
def load_data():
    # 载入真实标签 (提取最后一列)
    y_true = pd.read_csv(Config.LABEL_PATH).iloc[:, -1].values.ravel()

    # 载入各模型预测概率
    pred_dfs = {}
    for name, path in Config.MODEL_PREDS.items():
        if os.path.exists(path):
            # 获取 prob 列数值
            pred_dfs[name] = pd.read_csv(path)["prob"].values
        else:
            print(f"⚠️ 警告: 未找到 {name} 的预测文件，路径: {path}")

    return y_true, pred_dfs

# ==========================================
# 3. Optuna 寻优核心逻辑
# ==========================================
def objective(trial, y_true, pred_dfs):
    weights = {}
    for name in pred_dfs.keys():
        weights[name] = trial.suggest_float(name, 0.0, 1.0)

    # 归一化权重
    total_w = sum(weights.values())
    if total_w == 0: return 1.0 # 惩罚项

    blended_prob = np.zeros_like(y_true, dtype=float)
    for name, prob in pred_dfs.items():
        blended_prob += prob * (weights[name] / total_w)

    # 在 Top 3 任务中，我们希望 AUC 高且 LogLoss 低
    # 组合分数 = AUC - LogLoss (或者只看 AUC)
    auc = roc_auc_score(y_true, blended_prob)
    
    return auc

# ==========================================
# 4. 执行融合与可视化
# ==========================================
def run_blending():
    print(f"[{time.strftime('%H:%M:%S')}] 启动 Top 3 模型集成寻优...")
    y_true, pred_dfs = load_data()

    if len(pred_dfs) < 2:
        print("❌ 错误: 有效模型不足 2 个。")
        return

    # 1. 自动搜索最优权重分配
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, y_true, pred_dfs), n_trials=Config.N_TRIALS)

    # 2. 提取并归一化最佳权重
    best_raw = study.best_params
    total_best_w = sum(best_raw.values())
    best_weights = {k: round(v / total_best_w, 4) for k, v in best_raw.items()}

    # 3. 计算最终融合指标
    final_prob = np.zeros_like(y_true, dtype=float)
    for name, prob in pred_dfs.items():
        final_prob += prob * best_weights[name]
    
    final_auc = roc_auc_score(y_true, final_prob)
    final_loss = log_loss(y_true, final_prob)

    print("\n🏆 融合结果报告")
    print("=" * 40)
    for name, w in best_weights.items():
        print(f" 🟢 {name:10} 贡献权重: {w*100:>6.2f}%")
    print("-" * 40)
    print(f" 🚀 融合后最终 AUC: {final_auc:.6f}")
    print(f" 📉 融合后 LogLoss: {final_loss:.6f}")
    print("=" * 40)

    # 4. 保存结果
    res_df = pd.DataFrame({"prob": final_prob})
    res_df.to_csv(os.path.join(Config.OUTPUT_DIR, "final_blended_top3_preds.csv"), index=False)

    # 5. 权重占比饼图
    plot_weights(best_weights)

def plot_weights(weights):
    plt.figure(figsize=(10, 6), facecolor='white')
    names = list(weights.keys())
    vals = list(weights.values())
    
    # 使用更有质感的颜色
    colors = sns.color_palette("viridis", len(names))
    
    plt.pie(vals, labels=names, autopct="%1.1f%%", startangle=140, 
            colors=colors, explode=[0.03] * len(names), shadow=True)
    plt.title("Top 3 Ensemble: Optimized Model Contribution", fontsize=14)
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "top3_weights_pie.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_blending()