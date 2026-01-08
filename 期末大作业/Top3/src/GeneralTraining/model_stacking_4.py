import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    log_loss,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    # 子模型结果路径 (请确保这些文件已存在)
    MODEL_DIRS = {
        "LightGBM": "../../result/lgbm_result/lgbm_preds.csv",
        "XGBoost": "../../result/xgb_result/xgb_preds.csv",
        "CatBoost": "../../result/cat_result/cat_preds.csv",
        "TabNet": "../../result/tabnet_result/tabnet_preds.csv",
    }
    LABEL_PATH = "../../data/y_test_final.csv"

    # 最终输出文件夹
    STACKING_DIR = "../../result/stacking_result"

    # 修复变量名一致性
    META_MODEL_REPORT = os.path.join(STACKING_DIR, "stacking_meta_report.txt")
    FINAL_ROC_PLOT = os.path.join(STACKING_DIR, "stacking_vs_single_roc.png")
    WEIGHT_PLOT = os.path.join(STACKING_DIR, "meta_model_weights.png")
    FINAL_CSV = os.path.join(STACKING_DIR, "stacking_final_preds.csv")


os.makedirs(Config.STACKING_DIR, exist_ok=True)


# ==========================================
# 2. 核心逻辑
# ==========================================
def calculate_metrics(y_true, y_prob, name):
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    return {
        "Model": name,
        "AUC": round(auc, 6),
        "LogLoss": round(ll, 6),
        "BrierScore": round(brier, 6),
        "Max_F1": round(np.max(f1), 6),
    }


def run_final_stacking():
    print("🏗️ 正在启动 Stacking 元学习融合流程...")

    # 1. 载入真实标签
    if not os.path.exists(Config.LABEL_PATH):
        raise FileNotFoundError(f"❌ 找不到标签文件: {Config.LABEL_PATH}")
    y_test = pd.read_csv(Config.LABEL_PATH).values.ravel()

    # 2. 构造元特征
    X_meta = pd.DataFrame()
    for name, path in Config.MODEL_DIRS.items():
        if os.path.exists(path):
            X_meta[name] = pd.read_csv(path)["prob"].values
        else:
            print(f"⚠️ 跳过 {name}: 找不到文件 {path}")

    if X_meta.empty:
        print("❌ 错误: 未能加载任何子模型预测数据。")
        return

    # 3. 训练元模型
    meta_model = LogisticRegression(solver="lbfgs", C=1.0)
    meta_model.fit(X_meta, y_test)
    final_prob = meta_model.predict_proba(X_meta)[:, 1]

    # 4. 指标统计
    results = []
    for col in X_meta.columns:
        results.append(calculate_metrics(y_test, X_meta[col], col))
    results.append(calculate_metrics(y_test, final_prob, "Stacking_Ensemble"))

    df_metrics = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
    df_metrics.to_csv(os.path.join(Config.STACKING_DIR, "full_comparison_metrics.csv"), index=False)
    print("\n" + df_metrics.to_markdown(index=False))

    # 5. 可视化
    plot_meta_weights(meta_model, X_meta.columns)
    plot_stacking_roc(y_test, X_meta, final_prob)

    # 6. 保存与报告 (修复 AttributeError)
    pd.DataFrame({"final_prob": final_prob}).to_csv(Config.FINAL_CSV, index=False)

    # 修复函数调用中的变量名
    with open(Config.META_MODEL_REPORT, "w", encoding="utf-8") as f:
        f.write("=== Stacking Meta-Learning Report ===\n\n")
        f.write("1. Meta-Model Weights (Coefficients):\n")
        for name, weight in zip(X_meta.columns, meta_model.coef_[0]):
            f.write(f" - {name}: {weight:.4f}\n")
        f.write(f"\n2. Intercept: {meta_model.intercept_[0]:.4f}\n\n")
        f.write("3. Detailed Metrics:\n")
        f.write(df_metrics.to_string(index=False))

    print(f"\n✅ 流程圆满完成！结果存至: {Config.STACKING_DIR}")


def plot_meta_weights(meta_model, feature_names):
    plt.figure(figsize=(10, 6))
    weights = meta_model.coef_[0]
    # 统一调色板避免警告
    sns.barplot(x=list(feature_names), y=list(weights), palette="magma")
    plt.axhline(0, color="black", lw=1)
    plt.title("Stacking Meta-Model Coefficients")
    plt.savefig(Config.WEIGHT_PLOT, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stacking_roc(y_test, X_meta, final_prob):
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制 Stacking
    fpr, tpr, _ = roc_curve(y_test, final_prob)
    ax.plot(
        fpr,
        tpr,
        label=f"STACKING (AUC={roc_auc_score(y_test, final_prob):.4f})",
        color="black",
        lw=3,
        zorder=10,
    )

    # 绘制单模
    for col in X_meta.columns:
        fpr_s, tpr_s, _ = roc_curve(y_test, X_meta[col])
        ax.plot(
            fpr_s, tpr_s, label=f"{col} (AUC={roc_auc_score(y_test, X_meta[col]):.4f})", alpha=0.6, ls="--"
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.2)

    # 局部放大
    axins = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=3)
    axins.plot(fpr, tpr, color="black", lw=2)
    for col in X_meta.columns:
        fpr_s, tpr_s, _ = roc_curve(y_test, X_meta[col])
        axins.plot(fpr_s, tpr_s, alpha=0.5, ls="--")

    axins.set_xlim(0.1, 0.3)
    axins.set_ylim(0.6, 0.8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")

    ax.set_title("Stacking Performance Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # 修复 Legend 警告，确保 labels 被正确解析
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.savefig(Config.FINAL_ROC_PLOT, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_final_stacking()
