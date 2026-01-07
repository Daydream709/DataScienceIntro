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
# 1. 配置参数
# ==========================================
class Config:
    MODEL_DIRS = {
        "LightGBM": "../../result/lgbm_tuning_result/lgbm_tuned_preds.csv",
        "XGBoost": "../../result/xgb_tuning_result/xgb_tuned_preds.csv",
        "CatBoost": "../../result/cat_tuning_result/cat_tuned_preds.csv",
        "TabNet": "../../result/tabnet_result/tabnet_preds.csv",
    }
    LABEL_PATH = "../../data/y_test_final.csv"

    STACKING_DIR = "../../result/stacking_tuning_result"
    META_REPORT_TXT = os.path.join(STACKING_DIR, "stacking_meta_report.txt")
    DASHBOARD_PNG = os.path.join(STACKING_DIR, "stacking_dashboard.png")
    FINAL_CSV = os.path.join(STACKING_DIR, "stacking_final_preds.csv")


os.makedirs(Config.STACKING_DIR, exist_ok=True)


# ==========================================
# 2. 评估工具
# ==========================================
def calculate_metrics(y_true, y_prob, name):
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    # Brier Score 越小，说明预测概率的“诚实度”越高
    brier = brier_score_loss(y_true, y_prob)
    prec, rec, thres = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

    return {"Model": name, "AUC": auc, "LogLoss": ll, "Brier": brier, "Max_F1": np.max(f1)}


# ==========================================
# 3. 资产生成 (1x3 看板)
# ==========================================
def generate_stacking_assets(y_test, X_meta, final_prob, meta_model, metrics_df):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # 子图 1: ROC 曲线对比 (包含局部放大图)
    fpr_f, tpr_f, _ = roc_curve(y_test, final_prob)
    axes[0].plot(
        fpr_f,
        tpr_f,
        label=f"Stacking Final (AUC={metrics_df.loc[metrics_df['Model']=='Stacking_Final', 'AUC'].values[0]:.4f})",
        color="black",
        lw=3,
    )

    colors = sns.color_palette("husl", len(X_meta.columns))
    for i, col in enumerate(X_meta.columns):
        fpr_s, tpr_s, _ = roc_curve(y_test, X_meta[col])
        axes[0].plot(fpr_s, tpr_s, label=f"{col}", alpha=0.6, ls="--", color=colors[i])

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.2)
    axes[0].set_title("ROC Comparison: Stacking vs Singles", fontsize=14)
    axes[0].legend(loc="lower right")

    # 局部放大
    try:
        axins = inset_axes(axes[0], width="35%", height="35%", loc="center right", borderpad=2)
        axins.plot(fpr_f, tpr_f, color="black", lw=2)
        axins.set_xlim(0.05, 0.2)
        axins.set_ylim(0.6, 0.85)
        mark_inset(axes[0], axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")
    except:
        pass

    # 子图 2: PR 曲线 (关注长尾正样本的召回)
    prec_f, rec_f, _ = precision_recall_curve(y_test, final_prob)
    axes[1].plot(rec_f, prec_f, color="blue", lw=2, label="Stacking PR")
    axes[1].set_title("Precision-Recall (Stacking)", fontsize=14)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(alpha=0.3)

    # 子图 3: 元模型权重 (Coefficients)
    weights = meta_model.coef_[0]
    sns.barplot(x=list(X_meta.columns), y=list(weights), ax=axes[2], palette="viridis")
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_title("Meta-Model Coefficients (Influence)", fontsize=14)
    axes[2].set_ylabel("Weight Value")

    plt.tight_layout()
    plt.savefig(Config.DASHBOARD_PNG, dpi=300)
    plt.close()


# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 🏗️ 启动 Stacking 流程...")

    # 1. 载入标签
    y_test = pd.read_csv(Config.LABEL_PATH).iloc[:, -1].values.ravel()

    # 2. 构造元特征 (对齐特征列)
    X_meta = pd.DataFrame()
    for name, path in Config.MODEL_DIRS.items():
        if os.path.exists(path):
            X_meta[name] = pd.read_csv(path)["prob"].values
        else:
            print(f"⚠️ 跳过 {name}: 文件缺失")

    # 3. 训练 Meta-Learner
    # 逻辑回归作为元模型，可以看作是学习各模型的“可信度”
    meta_model = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2")
    meta_model.fit(X_meta, y_test)
    final_prob = meta_model.predict_proba(X_meta)[:, 1]

    # 4. 统计对比
    results = []
    for col in X_meta.columns:
        results.append(calculate_metrics(y_test, X_meta[col], col))
    results.append(calculate_metrics(y_test, final_prob, "Stacking_Final"))

    df_metrics = pd.DataFrame(results).sort_values(by="AUC", ascending=False)

    # 5. 生成资产
    generate_stacking_assets(y_test, X_meta, final_prob, meta_model, df_metrics)

    # 6. 归档预测结果与报告
    pd.DataFrame({"prob": final_prob}).to_csv(Config.FINAL_CSV, index=False)

    with open(Config.META_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 45 + "\n")
        f.write("      Stacking 元学习集成终极报告\n")
        f.write("=" * 45 + "\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("📊 [模型表现排行榜]\n")
        f.write(df_metrics.to_string(index=False))
        f.write("\n\n⚖️ [元学习器话语权分配]\n")
        for name, weight in zip(X_meta.columns, meta_model.coef_[0]):
            f.write(f" - {name:10}: {weight:.4f}\n")
        f.write(f"\n- 偏置项 (Bias): {meta_model.intercept_[0]:.4f}\n")

    print(f"\n✅ 任务完成！终极看板: {Config.DASHBOARD_PNG}")
    print(f"📈 最终融合 AUC: {df_metrics.loc[df_metrics['Model']=='Stacking_Final', 'AUC'].values[0]:.6f}")


if __name__ == "__main__":
    main()
