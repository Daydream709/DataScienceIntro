import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score, f1_score, log_loss, roc_curve, precision_recall_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

warnings.filterwarnings("ignore")


# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    TRAIN_FEAT_PATH = "../../data/X_train_final.csv"
    TRAIN_LABEL_PATH = "../../data/y_train_final.csv"
    TEST_FEAT_PATH = "../../data/X_test_final.csv"
    TEST_LABEL_PATH = "../../data/y_test_final.csv"

    RESULT_DIR = "../../result/tabnet_result"
    MODEL_SAVE_PATH = os.path.join(RESULT_DIR, "tabnet_model")
    PREDS_CSV = os.path.join(RESULT_DIR, "tabnet_preds.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "tabnet_dashboard.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "tabnet_evaluation_report.txt")

    # TabNet 核心超参数 (针对表格数据优化的深度学习)
    PARAMS = {
        "n_d": 32,
        "n_a": 32,
        "n_steps": 5,
        "gamma": 1.5,
        "n_independent": 2,
        "n_shared": 2,
        "momentum": 0.02,
        "clip_value": 2.0,
        "lambda_sparse": 1e-4,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=2e-2),
        "scheduler_params": {"step_size": 50, "gamma": 0.9},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "mask_type": "entmax",  # 比 softmax 更稀疏，适合特征选择
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
    }


os.makedirs(Config.RESULT_DIR, exist_ok=True)


# ==========================================
# 2. 评估资产生成 (1x3 看板)
# ==========================================
def generate_assets(y_test, y_prob, model, feature_names, params):
    # 1. 计算指标
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    prec, rec, thres = precision_recall_curve(y_test, y_prob)

    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    ix = np.argmax(f1)
    max_f1 = f1[ix]
    best_th = thres[ix] if ix < len(thres) else 0.5

    # 2. 特征重要性
    feat_imp = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": feat_imp}).sort_values(
        by="importance", ascending=False
    )

    # 3. 写入文本报告
    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write("      TabNet 深度学习评估报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"📈 [性能指标]\n")
        f.write(f" - AUC      : {auc:.6f}\n")
        f.write(f" - LogLoss  : {loss:.6f}\n")
        f.write(f" - Max F1   : {max_f1:.6f}\n")
        f.write(f" - Best Th  : {best_th:.4f}\n")
        f.write(f"\n📊 [Top 20 特征排名]\n")
        for i, row in imp_df.head(20).reset_index(drop=True).iterrows():
            f.write(f" {i+1:2}. {row['feature']:25}: {row['importance']:.6f}\n")

    # 4. 绘图 1x3
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # 子图 1: ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f"TabNet (AUC={auc:.4f})", color="#9467bd", lw=3)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC Curve", fontsize=14)
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # 子图 2: PR
    axes[1].plot(rec, prec, color="purple", lw=3, label=f"Max F1={max_f1:.3f}")
    axes[1].set_title("Precision-Recall Curve", fontsize=14)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 子图 3: Feature Importance
    top_20 = imp_df.head(20).sort_values(by="importance", ascending=True)
    axes[2].barh(top_20["feature"], top_20["importance"], color="#9467bd")
    axes[2].set_title("Top 20 Feature Importance", fontsize=14)
    axes[2].set_xlabel("Importance Score")
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# 3. 主程序
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 正在加载并清洗数据...")

    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    # 1. 彻底剔除非特征列 (物理隔离赔率和标签)
    drop_cols = ["raw_win_odds", "actual_rank", "race_id", "win_odds"]
    X_train_df = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_test_raw = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    # 2. 强制特征对齐
    feature_names = X_train_df.columns.tolist()
    X_test_df = X_test_raw.reindex(columns=feature_names, fill_value=0)

    # 3. 准备标签 (只取最后一列)
    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    # 4. 转换为 Numpy 格式 (TabNet 必需)
    X_train_np = X_train_df.values.astype(np.float32)
    X_test_np = X_test_df.values.astype(np.float32)

    print(f"✅ 对齐完成 | 特征数: {X_train_np.shape[1]} | 样本数: {len(X_train_np)}")

    # 5. 初始化 TabNet
    clf = TabNetClassifier(**Config.PARAMS)

    print(f"🚀 开始训练 TabNet (Device: {Config.PARAMS['device_name']})...")

    # 针对不平衡数据的权重处理 (可选)
    # weights = 1 为不加权。对于 1:10 的数据，TabNet 通常靠内部注意力机制处理，
    # 如果效果不好，可以在这里加 weights 参数。

    clf.fit(
        X_train=X_train_np,
        y_train=y_train,
        eval_set=[(X_test_np, y_test)],  # 重点关注验证集表现
        eval_name=["valid"],
        eval_metric=["auc"],
        max_epochs=200,
        patience=30,
        batch_size=16384,
        virtual_batch_size=1024,
        num_workers=0,
        drop_last=False,
    )

    # 6. 预测与保存
    y_prob = clf.predict_proba(X_test_np)[:, 1]
    pd.DataFrame({"prob": y_prob}).to_csv(Config.PREDS_CSV, index=False)

    # 7. 生成资产与保存模型
    generate_assets(y_test, y_prob, clf, feature_names, Config.PARAMS)

    # TabNet 自身的模型保存
    saved_filepath = clf.save_model(Config.MODEL_SAVE_PATH)

    print(f"\n✨ TabNet 流程完成!")
    print(f"📊 评估面板: {Config.PLOT_PNG}")
    print(f"💾 模型路径: {saved_filepath}")


if __name__ == "__main__":
    main()
