import os
import time
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve
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
    
    # 报告文件路径
    REPORT_TXT = os.path.join(OUTPUT_DIR, "blending_top3_report.txt")


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
# ==========================================
# 5. 计算F1分数和最佳阈值
# ==========================================
def calculate_f1_and_threshold(y_true, y_prob):
    # 计算精确率、召回率和阈值
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # 计算F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    # 找到最大F1分数和对应的阈值
    best_f1 = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    return best_f1, best_threshold

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
    
    # 计算各指标
    final_auc = roc_auc_score(y_true, final_prob)
    final_loss = log_loss(y_true, final_prob)
    final_f1, final_threshold = calculate_f1_and_threshold(y_true, final_prob)

    print("\n🏆 融合结果报告")
    print("=" * 40)
    for name, w in best_weights.items():
        print(f" 🟢 {name:10} 贡献权重: {w*100:>6.2f}%")
    print("-" * 40)
    print(f" 🚀 融合后最终 AUC: {final_auc:.6f}")
    print(f" 📉 融合后 LogLoss: {final_loss:.6f}")
    print(f" 🎯 融合后最佳 F1: {final_f1:.6f}")
    print(f" 🎯 最佳分类阈值: {final_threshold:.4f}")
    print("=" * 40)

    # 4. 保存结果
    res_df = pd.DataFrame({"prob": final_prob})
    res_df.to_csv(os.path.join(Config.OUTPUT_DIR, "final_blended_top3_preds.csv"), index=False)

    # 5. 生成报告文件
    generate_report(y_true, pred_dfs, best_weights, final_prob, final_auc, final_loss, final_f1, final_threshold)

    # 6. 权重占比饼图
    plot_weights(best_weights)
    
    # 7. 模型相关性热力图
    plot_correlation_heatmap(pred_dfs)

# ==========================================
# 6. 生成报告文件
# ==========================================
def generate_report(y_true, pred_dfs, best_weights, final_prob, final_auc, final_loss, final_f1, final_threshold):
    # 计算各单一模型的指标
    model_metrics = []
    for name, prob in pred_dfs.items():
        auc = roc_auc_score(y_true, prob)
        loss = log_loss(y_true, prob)
        f1, threshold = calculate_f1_and_threshold(y_true, prob)
        model_metrics.append({
            "Model": name,
            "AUC": auc,
            "F1_score": f1,
            "LogLoss": loss,
            "Best_Threshold": threshold
        })
    
    # 添加融合模型的指标
    model_metrics.append({
        "Model": "Blending",
        "AUC": final_auc,
        "F1_score": final_f1,
        "LogLoss": final_loss,
        "Best_Threshold": final_threshold
    })
    
    # 转换为DataFrame以便排序
    df_metrics = pd.DataFrame(model_metrics).sort_values(by="AUC", ascending=False)
    
    # 写入报告文件
    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=============================================\n")
        f.write("      Top 3 模型集成(Blending)报告\n")
        f.write("=============================================\n")
        f.write(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 性能对比排行榜
        f.write("📊 [性能对比排行榜]\n")
        f.write(" 模型名称      AUC      | F1 分数  | LogLoss  | Best Threshold\n")
        f.write("-" * 65 + "\n")
        for _, row in df_metrics.iterrows():
            f.write(f" {row['Model']:10}  {row['AUC']:.6f} | {row['F1_score']:.6f} | {row['LogLoss']:.6f} | {row['Best_Threshold']:.4f}\n")
        
        # 最优权重分配
        f.write("\n⚖️ [最优权重分配]\n")
        for name, w in best_weights.items():
            f.write(f" - {name:10}: {w*100:.2f}%\n")

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

# ==========================================
# 8. 模型相关性热力图
# ==========================================
def plot_correlation_heatmap(pred_dfs):
    # 将模型预测结果转换为DataFrame
    pred_df = pd.DataFrame(pred_dfs)
    
    # 计算皮尔逊相关系数
    correlation_matrix = pred_df.corr()
    
    # 绘制热力图
    plt.figure(figsize=(10, 8), facecolor='white')
    
    # 使用seaborn绘制热力图，添加数值标签
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".4f",
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        annot_kws={"size": 12}
    )
    
    plt.title("Top 3 Models: Prediction Correlation Heatmap", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    
    # 保存热力图
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "top3_model_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_blending()