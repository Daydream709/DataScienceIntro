import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ==========================================
# 1. 配置参数 (对接 Top 3 调参后的路径)
# ==========================================
class Config:
    # 自动对应你之前 Top 3 调参脚本生成的路径
    MODEL_DIRS = {
        "LightGBM": "../../result/lgbm_top3_tuning_result/lgbm_tuned_preds_top3.csv",
        "XGBoost": "../../result/xgb_top3_tuning_result/xgb_tuned_preds_top3.csv",
        "CatBoost": "../../result/cat_top3_tuning_result/cat_tuned_preds_top3.csv",
        "TabNet": "../../result/tabnet_top3_result/tabnet_preds_top3.csv",
    }
    LABEL_PATH = "../../data/y_test_final_top3.csv"

    STACKING_DIR = "../../result/stacking_top3_result"
    META_MODEL_REPORT = os.path.join(STACKING_DIR, "stacking_meta_report_top3.txt")
    FINAL_ROC_PLOT = os.path.join(STACKING_DIR, "stacking_vs_single_roc_top3.png")
    WEIGHT_PLOT = os.path.join(STACKING_DIR, "meta_model_weights_top3.png")
    FINAL_CSV = os.path.join(STACKING_DIR, "stacking_final_preds_top3.csv")

os.makedirs(Config.STACKING_DIR, exist_ok=True)

# ==========================================
# 2. 核心评估函数
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

# ==========================================
# 3. 主程序
# ==========================================
def run_final_stacking():
    print(f"[{time.strftime('%H:%M:%S')}] 🏗️ 启动 Top 3 Stacking 元学习融合...")

    # 1. 载入真实标签 (Top 3 标签在最后一列)
    if not os.path.exists(Config.LABEL_PATH):
        raise FileNotFoundError(f"❌ 找不到标签文件: {Config.LABEL_PATH}")
    y_test = pd.read_csv(Config.LABEL_PATH).iloc[:, -1].values.ravel()

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

    # 3. 训练元模型 (Logistic Regression)
    # 在 Stacking 中，逻辑回归通过学习权重，将概率转化为更稳健的预测
    meta_model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
    meta_model.fit(X_meta, y_test)

    # 得到 Stacking 后的终极概率
    final_prob = meta_model.predict_proba(X_meta)[:, 1]

    # 4. 指标统计与对比
    results = []
    for col in X_meta.columns:
        results.append(calculate_metrics(y_test, X_meta[col], col))
    results.append(calculate_metrics(y_test, final_prob, "Stacking_Final"))

    df_metrics = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
    
    # 5. 可视化
    plot_meta_weights(meta_model, X_meta.columns)
    plot_stacking_roc(y_test, X_meta, final_prob)

    # 6. 保存预测结果与报告
    pd.DataFrame({"prob": final_prob}).to_csv(Config.FINAL_CSV, index=False)

    with open(Config.META_MODEL_REPORT, "w", encoding="utf-8") as f:
        f.write("=== Top 3 Stacking Meta-Learning Report ===\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("1. 模型权力分配 (Meta-Model Coefficients):\n")
        f.write("注：正数代表该模型对最终预测有正向贡献\n")
        for name, weight in zip(X_meta.columns, meta_model.coef_[0]):
            f.write(f" - {name:10}: {weight:.4f}\n")
        f.write(f"\n2. 截距 (Intercept): {meta_model.intercept_[0]:.4f}\n\n")
        f.write("3. 性能对比排行榜:\n")
        f.write(df_metrics.to_string(index=False))

    print("\n📊 终极性能对比 (AUC 降序):")
    print(df_metrics.to_string(index=False))
    print(f"\n✨ Top 3 终极融合完成！归档目录: {Config.STACKING_DIR}")

def plot_meta_weights(meta_model, feature_names):
    plt.figure(figsize=(10, 6))
    weights = meta_model.coef_[0]
    # 使用条形图显示各模型在元模型中的系数
    sns.barplot(x=list(feature_names), y=list(weights), palette="magma")
    plt.axhline(0, color="black", lw=1)
    plt.title("Meta-Model Coefficients: Who has the final say?")
    plt.ylabel("Coefficient Value")
    plt.savefig(Config.WEIGHT_PLOT, dpi=300, bbox_inches="tight")
    plt.close()

def plot_stacking_roc(y_test, X_meta, final_prob):
    fig, ax = plt.subplots(figsize=(11, 8))

    # 绘制 Stacking 曲线
    fpr, tpr, _ = roc_curve(y_test, final_prob)
    ax.plot(fpr, tpr, label=f"Stacking Final (AUC={roc_auc_score(y_test, final_prob):.4f})", 
            color="#2c3e50", lw=4, zorder=5)

    # 绘制各个子模型曲线
    colors = sns.color_palette("Set2", len(X_meta.columns))
    for i, col in enumerate(X_meta.columns):
        fpr_s, tpr_s, _ = roc_curve(y_test, X_meta[col])
        ax.plot(fpr_s, tpr_s, label=f"{col} (AUC={roc_auc_score(y_test, X_meta[col]):.4f})", 
                alpha=0.6, ls="--", color=colors[i])

    ax.plot([0, 1], [0, 1], "k--", alpha=0.2)
    ax.set_title("Top 3 Ensemble Performance: Stacking vs Single Models", fontsize=14)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    # 局部放大：高分马匹区间 (TPR在0.7-0.9之间)
    axins = inset_axes(ax, width="35%", height="35%", loc="center right", borderpad=2)
    axins.plot(fpr, tpr, color="#2c3e50", lw=2)
    axins.set_xlim(0.05, 0.25)
    axins.set_ylim(0.65, 0.85)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")

    plt.savefig(Config.FINAL_ROC_PLOT, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run_final_stacking()