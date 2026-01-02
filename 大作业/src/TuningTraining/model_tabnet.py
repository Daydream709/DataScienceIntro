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

    # 统一归档文件夹
    RESULT_DIR = "../../result/tabnet_result"
    MODEL_SAVE_PATH = os.path.join(RESULT_DIR, "tabnet_model")
    PREDS_CSV = os.path.join(RESULT_DIR, "tabnet_preds.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "tabnet_metrics_dashboard.png")
    IMPORTANCE_PNG = os.path.join(RESULT_DIR, "tabnet_feature_importance.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "tabnet_evaluation_report.txt")

    # TabNet 核心超参数
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
        "mask_type": "entmax",
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
    }


os.makedirs(Config.RESULT_DIR, exist_ok=True)


# ==========================================
# 2. 评估资产生成
# ==========================================
def generate_assets(y_test, y_prob, model, feature_names):
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    prec, rec, thres = precision_recall_curve(y_test, y_prob)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    max_f1, best_th = np.max(f1), thres[np.argmax(f1)]

    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"=== TabNet 统一评估报告 ===\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"AUC: {auc:.6f}\n")
        f.write(f"LogLoss: {loss:.6f}\n")
        f.write(f"Max F1: {max_f1:.6f}\n")
        f.write(f"Best Thresh: {best_th:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f"TabNet (AUC = {auc:.4f})", color="#9467bd", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)

    try:
        axins = inset_axes(ax1, width="40%", height="40%", loc="lower right", borderpad=3)
        axins.plot(fpr, tpr, color="#9467bd", lw=2)
        axins.set_xlim(0.1, 0.3)
        axins.set_ylim(0.6, 0.8)
        mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")
    except:
        pass
    ax1.set_title("ROC Curve")
    ax1.legend()

    ax2.plot(rec, prec, color="purple", lw=2, label=f"F1 Max = {max_f1:.4f}")
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    plt.savefig(Config.PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    feat_imp = model.feature_importances_
    idx = np.argsort(feat_imp)[-20:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), feat_imp[idx], align="center", color="#9467bd")
    plt.yticks(range(20), [feature_names[i] for i in idx])
    plt.title("Top 20 Features (TabNet Importance)")
    plt.savefig(Config.IMPORTANCE_PNG, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# 3. 主程序
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 正在加载并对齐数据...")

    # 读取原始数据
    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    # 1. 特征处理：训练集全量，测试集切片
    X_train_df = df_train.copy()
    X_test_raw = df_test.iloc[:, :-2].copy()

    # 2. 强制特征对齐
    feature_names = X_train_df.columns.tolist()
    X_test_df = X_test_raw.reindex(columns=feature_names, fill_value=0)

    # 3. 准备标签 (只取最后一列)
    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    # 4. 转换为 TabNet 要求的 Numpy 格式
    X_train_np = X_train_df.values.astype(np.float32)
    X_test_np = X_test_df.values.astype(np.float32)

    print(f"对齐完成。训练特征数: {X_train_np.shape[1]} | 验证特征数: {X_test_np.shape[1]}")

    # 5. 初始化 TabNet
    clf = TabNetClassifier(**Config.PARAMS)

    print("🚀 开始训练 TabNet 深度神经网络...")
    clf.fit(
        X_train=X_train_np,
        y_train=y_train,
        eval_set=[(X_train_np, y_train), (X_test_np, y_test)],
        eval_name=["train", "valid"],
        eval_metric=["auc"],
        max_epochs=200,
        patience=30,
        batch_size=16384,
        virtual_batch_size=1024,
        num_workers=0,
        drop_last=False,
    )

    # 6. 预测与保存 (注意 y_prob 的实际标签对齐)
    y_prob = clf.predict_proba(X_test_np)[:, 1]
    pd.DataFrame({"prob": y_prob, "actual": y_test}).to_csv(Config.PREDS_CSV, index=False)

    # 7. 生成资产与保存模型
    generate_assets(y_test, y_prob, clf, feature_names)
    clf.save_model(Config.MODEL_SAVE_PATH)

    print(f"✅ TabNet 流程完成，结果存放在: {Config.RESULT_DIR}")


if __name__ == "__main__":
    main()
