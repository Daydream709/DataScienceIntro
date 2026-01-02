import os
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    # 调参后的四个模型预测结果路径
    MODEL_PREDS = {
        "CatBoost": "../../result/cat_tuning_result/cat_tuned_preds.csv",
        "LightGBM": "../../result/lgbm_tuning_result/lgbm_tuned_preds.csv",
        "XGBoost": "../../result/xgb_tuning_result/xgb_tuned_preds.csv",
        "TabNet": "../../result/tabnet_result/tabnet_preds.csv",
    }

    # 标签路径 (假设现在只有一列 Label)
    LABEL_PATH = "../../data/y_test_final.csv"

    # 输出目录
    OUTPUT_DIR = "../../result/blending_tuning_result"
    N_TRIALS = 100  # 权重寻优迭代次数


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. 载入数据
# ==========================================
def load_data():
    # 载入真实标签 (无需剔除，直接平铺)
    y_true = pd.read_csv(Config.LABEL_PATH).values.ravel()

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
    # 为每个模型建议一个 0.0 到 1.0 之间的权重
    for name in pred_dfs.keys():
        weights[name] = trial.suggest_float(name, 0.0, 1.0)

    # 权重归一化 (Softmax 思想：确保总和为 1)
    total_w = sum(weights.values())
    if total_w == 0:
        return 0

    # 计算加权融合后的最终概率
    blended_prob = np.zeros_like(y_true, dtype=float)
    for name, prob in pred_dfs.items():
        blended_prob += prob * (weights[name] / total_w)

    # 优化目标：最大化 AUC
    return roc_auc_score(y_true, blended_prob)


# ==========================================
# 4. 执行融合
# ==========================================
def run_blending():
    print("🚀 启动全模型最优权重搜索 (CatBoost + LGBM + XGB + TabNet)...")
    y_true, pred_dfs = load_data()

    if len(pred_dfs) < 2:
        print("❌ 错误: 有效模型不足 2 个，无法进行融合。")
        return

    # 1. 自动搜索最优权重分配
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, y_true, pred_dfs), n_trials=Config.N_TRIALS)

    # 2. 提取并归一化最佳权重
    best_raw = study.best_params
    total_best_w = sum(best_raw.values())
    best_weights = {k: round(v / total_best_w, 4) for k, v in best_raw.items()}

    print("\n🏆 寻优结束！")
    print("-" * 30)
    for name, w in best_weights.items():
        print(f" 模型: {name:10} | 最优权重: {w:.4f}")
    print("-" * 30)
    print(f"✨ 融合后最终 AUC: {study.best_value:.6f}")

    # 3. 生成并保存最终结果
    final_prob = np.zeros_like(y_true, dtype=float)
    for name, prob in pred_dfs.items():
        final_prob += prob * best_weights[name]

    pd.DataFrame({"prob": final_prob}).to_csv(
        os.path.join(Config.OUTPUT_DIR, "final_blended_preds.csv"), index=False
    )

    # 4. 权重占比可视化
    plot_weights(best_weights)

    print(f"\n✅ 终极融合预测已保存至: {Config.OUTPUT_DIR}")


def plot_weights(weights):
    plt.figure(figsize=(10, 6))
    names = list(weights.keys())
    vals = list(weights.values())

    colors = sns.color_palette("pastel")[0 : len(names)]
    plt.pie(vals, labels=names, autopct="%1.1f%%", startangle=140, colors=colors, explode=[0.05] * len(names))
    plt.title("Optimized Model Contribution (Ensemble Weight Distribution)")
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "optimized_weights_pie.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    run_blending()
