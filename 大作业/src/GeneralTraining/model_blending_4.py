import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, log_loss, roc_curve, precision_recall_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ==========================================
# 1. 配置参数 - 实验收割模式
# ==========================================
class Config:
    # 子模型结果路径
    MODEL_DIRS = {
        "LightGBM": "../../result/lgbm_result/lgbm_preds.csv",
        "XGBoost": "../../result/xgb_result/xgb_preds.csv",
        "CatBoost": "../../result/cat_result/cat_preds.csv",
        "TabNet": "../../result/tabnet_result/tabnet_preds.csv",
    }
    LABEL_PATH = "../../data/y_test_final.csv"

    # 最终输出文件夹
    FINAL_DIR = "../../result/blending_result"

    # 融合权重 (根据各模型单模表现调整，CatBoost最高)
    WEIGHTS = {"LightGBM": 0.25, "XGBoost": 0.15, "CatBoost": 0.40, "TabNet": 0.20}


os.makedirs(Config.FINAL_DIR, exist_ok=True)


# ==========================================
# 2. 核心逻辑：融合与评估
# ==========================================
def run_final_blending():
    print("🔮 正在启动全模型标准化融合流程...")

    # 1. 载入真实标签
    y_test = pd.read_csv(Config.LABEL_PATH).values.ravel()

    # 2. 载入各模型预测概率并构建 DataFrame
    all_probs = pd.DataFrame()
    for name, path in Config.MODEL_DIRS.items():
        if os.path.exists(path):
            all_probs[name] = pd.read_csv(path)["prob"].values
        else:
            print(f"⚠️ 警告: 未找到 {name} 的预测文件，请检查路径。")

    # 3. 计算 Blending 概率
    final_prob = np.zeros_like(y_test, dtype=float)
    for name, weight in Config.WEIGHTS.items():
        final_prob += all_probs[name] * weight
    all_probs["Blending_Ensemble"] = final_prob

    # 4. 指标统计
    metrics_list = []
    for col in all_probs.columns:
        prob = all_probs[col]
        auc = roc_auc_score(y_test, prob)
        loss = log_loss(y_test, prob)
        prec, rec, _ = precision_recall_curve(y_test, prob)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        metrics_list.append(
            {"Model": col, "AUC": round(auc, 6), "LogLoss": round(loss, 6), "Max_F1": round(np.max(f1), 6)}
        )

    df_metrics = pd.DataFrame(metrics_list).sort_values(by="AUC", ascending=False)

    # 保存指标表格
    df_metrics.to_csv(os.path.join(Config.FINAL_DIR, "overall_metrics.csv"), index=False)
    print("\n" + df_metrics.to_markdown(index=False))

    # 5. 可视化 A：全模型 ROC 对比图 (带局部放大)
    plot_comparison_roc(y_test, all_probs)

    # 6. 可视化 B：模型相关性热力图 (冗余分析)
    plot_correlation(all_probs.drop(columns="Blending_Ensemble"))

    # 7. 保存最终预测结果
    all_probs[["Blending_Ensemble"]].to_csv(
        os.path.join(Config.FINAL_DIR, "final_ensemble_preds.csv"), index=False
    )
    print(f"\n✅ 终极报告已生成至: {Config.FINAL_DIR}")


# ==========================================
# 3. 绘图函数库
# ==========================================
def plot_comparison_roc(y_test, df_probs):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    colors = sns.color_palette("husl", len(df_probs.columns))

    for i, col in enumerate(df_probs.columns):
        fpr, tpr, _ = roc_curve(y_test, df_probs[col])
        lw = 3 if col == "Blending_Ensemble" else 1.5
        alpha = 1.0 if col == "Blending_Ensemble" else 0.7
        linestyle = "-" if col == "Blending_Ensemble" else "--"
        color = "black" if col == "Blending_Ensemble" else colors[i]

        plt.plot(
            fpr,
            tpr,
            label=f"{col} (AUC={roc_auc_score(y_test, df_probs[col]):.4f})",
            lw=lw,
            alpha=alpha,
            linestyle=linestyle,
            color=color,
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.2)

    # 局部放大
    axins = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=3)
    for i, col in enumerate(df_probs.columns):
        fpr, tpr, _ = roc_curve(y_test, df_probs[col])
        color = "black" if col == "Blending_Ensemble" else colors[i]
        axins.plot(fpr, tpr, color=color, lw=2 if col == "Blending_Ensemble" else 1)

    axins.set_xlim(0.1, 0.3)
    axins.set_ylim(0.6, 0.8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")

    plt.title("Final Model Comparison: ROC Curves", fontsize=15)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(Config.FINAL_DIR, "final_roc_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation(df_probs):
    plt.figure(figsize=(10, 8))
    corr = df_probs.corr()
    sns.heatmap(corr, annot=True, cmap="RdYlGn", fmt=".4f", center=0.95)
    plt.title("Model Prediction Correlation (Redundancy Analysis)", fontsize=15)
    plt.savefig(os.path.join(Config.FINAL_DIR, "model_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_final_blending()
