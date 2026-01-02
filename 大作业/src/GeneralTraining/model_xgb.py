import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, log_loss, roc_curve, precision_recall_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 忽略警告
warnings.filterwarnings("ignore")


# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    # 确保路径指向你最新的生成文件
    TRAIN_FEAT_PATH = "../../data/X_train_final.csv"
    TRAIN_LABEL_PATH = "../../data/y_train_final.csv"
    TEST_FEAT_PATH = "../../data/X_test_final.csv"
    TEST_LABEL_PATH = "../../data/y_test_final.csv"

    RESULT_DIR = "../../result/xgb_result"
    MODEL_PKL = os.path.join(RESULT_DIR, "xgb_model.pkl")
    PREDS_CSV = os.path.join(RESULT_DIR, "xgb_preds.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "xgb_metrics_dashboard.png")
    IMPORTANCE_PNG = os.path.join(RESULT_DIR, "xgb_feature_importance.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "xgb_evaluation_report.txt")

    PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "device": "cuda",  # 如果没有显卡请改为 "cpu"
        "learning_rate": 0.005,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "scale_pos_weight": 4.0,
        "alpha": 1.0,  # L1 正则
        "lambda": 1.0,  # L2 正则
        "n_jobs": -1,
    }
    NUM_ROUNDS = 10000
    EARLY_STOP = 200


# 创建输出目录
os.makedirs(Config.RESULT_DIR, exist_ok=True)


def log_print(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ==========================================
# 2. 评估资产生成
# ==========================================
def generate_assets(y_test, y_prob, model):
    log_print("正在生成评估资产...")
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    prec, rec, thres = precision_recall_curve(y_test, y_prob)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    max_f1, best_th = np.max(f1), thres[np.argmax(f1)]

    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"=== XGBoost 统一评估报告 ===\n")
        f.write(f"AUC: {auc:.6f}\nLogLoss: {loss:.6f}\nMax F1: {max_f1:.6f}\nBest Thresh: {best_th:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f"XGB (AUC = {auc:.4f})", color="#ff7f0e", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)

    try:
        axins = inset_axes(ax1, width="40%", height="40%", loc="lower right", borderpad=3)
        axins.plot(fpr, tpr, color="#ff7f0e", lw=2)
        axins.set_xlim(0.1, 0.3)
        axins.set_ylim(0.6, 0.8)
        mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")
    except:
        pass

    ax2.plot(rec, prec, color="darkorange", lw=2, label=f"F1 Max = {max_f1:.4f}")
    ax2.set_title("PR Curve")
    plt.savefig(Config.PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=20, importance_type="gain", ax=plt.gca(), color="orange")
    plt.title("Top 20 Features (XGBoost Gain)")
    plt.savefig(Config.IMPORTANCE_PNG, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# 3. 主程序
# ==========================================
def main():
    log_print("正在读取并对齐数据...")
    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    # 1. 差异化提取特征
    # 训练集：直接作为特征
    X_train = df_train.copy()

    # 测试集：严格剔除最后两列 (raw_win_odds, actual_rank)
    X_test_temp = df_test.iloc[:, :-2].copy()

    # 2. 强制对齐特征列名
    feature_names = X_train.columns.tolist()
    X_test = X_test_temp.reindex(columns=feature_names, fill_value=0)

    # 3. 加载标签
    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    log_print(f"对齐完成。训练特征数: {X_train.shape[1]} | 测试特征数: {X_test.shape[1]}")

    # 4. 构建 XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    # 5. 训练
    log_print("开始 XGBoost 训练...")
    model = xgb.train(
        Config.PARAMS,
        dtrain,
        num_boost_round=Config.NUM_ROUNDS,
        evals=[(dtrain, "train"), (dtest, "valid")],
        early_stopping_rounds=Config.EARLY_STOP,
        verbose_eval=100,
    )

    # 6. 预测与保存
    y_prob = model.predict(dtest)
    pd.DataFrame({"prob": y_prob, "actual": y_test}).to_csv(Config.PREDS_CSV, index=False)

    generate_assets(y_test, y_prob, model)

    with open(Config.MODEL_PKL, "wb") as f:
        pickle.dump(model, f)

    log_print(f"✅ 任务完成！结果目录: {Config.RESULT_DIR}")


if __name__ == "__main__":
    main()
