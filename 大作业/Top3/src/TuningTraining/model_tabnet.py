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
# 1. 配置参数 (更新为 Top 3 路径)
# ==========================================
class Config:
    TRAIN_FEAT_PATH = "../../data/X_train_final_top3.csv"
    TRAIN_LABEL_PATH = "../../data/y_train_final_top3.csv"
    TEST_FEAT_PATH = "../../data/X_test_final_top3.csv"
    TEST_LABEL_PATH = "../../data/y_test_final_top3.csv"

    RESULT_DIR = "../../result/tabnet_top3_result"
    MODEL_SAVE_PATH = os.path.join(RESULT_DIR, "tabnet_model_top3")
    PREDS_CSV = os.path.join(RESULT_DIR, "tabnet_preds_top3.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "tabnet_metrics_dashboard_top3.png")
    IMPORTANCE_PNG = os.path.join(RESULT_DIR, "tabnet_feature_importance_top3.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "tabnet_evaluation_report_top3.txt")

    # TabNet 核心参数调整
    PARAMS = {
        "n_d": 32,
        "n_a": 32,
        "n_steps": 4,          # 前三名预测任务稍减步骤，防止过拟合噪声
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "momentum": 0.02,
        "clip_value": 2.0,
        "lambda_sparse": 1e-4,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=1e-2), # 学习率微降，提高稳定性
        "scheduler_params": {"step_size": 30, "gamma": 0.9},
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
        f.write(f"=== TabNet Top 3 评估报告 ===\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"AUC: {auc:.6f} | LogLoss: {loss:.6f}\n")
        f.write(f"Max F1: {max_f1:.6f} | Best Thresh: {best_th:.4f}\n")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f"TabNet Top 3 (AUC = {auc:.4f})", color="#9467bd", lw=3)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PR Curve
    ax2.plot(rec, prec, color="purple", lw=3, label=f"Top 3 PR Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 特征重要性
    feat_imp = model.feature_importances_
    idx = np.argsort(feat_imp)[-20:]
    ax3.barh(range(20), feat_imp[idx], align="center", color="#9b59b6")
    ax3.set_yticks(range(20))
    ax3.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax3.set_title("Top 20 Features (TabNet Top 3)")
    ax3.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(Config.PLOT_PNG, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. 主程序
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 加载数据并对齐特征...")

    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    # 1. 特征切片：剔除末尾三列 (raw_win_odds, actual_rank, race_id)
    X_train_df = df_train.copy()
    X_test_raw = df_test.iloc[:, :-3].copy() 

    feature_names = X_train_df.columns.tolist()
    X_test_df = X_test_raw.reindex(columns=feature_names, fill_value=0)

    # 2. 标签加载
    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    # 3. 权重平衡 (针对 21.63% 正样本计算权重)
    # TabNet 通过 weights 参数指定每个样本的权重
    weights = np.where(y_train == 1, 3.6, 1.0) # 1:(78/22) 约等于 1:3.6

    X_train_np = X_train_df.values.astype(np.float32)
    X_test_np = X_test_df.values.astype(np.float32)

    print(f"训练集正样本率: {np.mean(y_train):.2%}")

    clf = TabNetClassifier(**Config.PARAMS)

    print("🚀 启动 TabNet Top 3 训练...")
    clf.fit(
        X_train=X_train_np,
        y_train=y_train,
        weights=1, # 也可以设置为 1 并让模型内部平衡，或者传入 weights 数组
        eval_set=[(X_test_np, y_test)],
        eval_name=["valid"],
        eval_metric=["auc"],
        max_epochs=200,
        patience=40,
        batch_size=16384,
        virtual_batch_size=1024,
        num_workers=0,
        drop_last=False,
        loss_fn=torch.nn.functional.cross_entropy # 显式指定交叉熵
    )

    y_prob = clf.predict_proba(X_test_np)[:, 1]
    pd.DataFrame({"prob": y_prob, "actual": y_test}).to_csv(Config.PREDS_CSV, index=False)

    generate_assets(y_test, y_prob, clf, feature_names)
    clf.save_model(Config.MODEL_SAVE_PATH)

    print(f"✅ TabNet Top 3 流程圆满完成！")

if __name__ == "__main__":
    main()