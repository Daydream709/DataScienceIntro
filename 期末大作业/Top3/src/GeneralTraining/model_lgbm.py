import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, log_loss, roc_curve, precision_recall_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 忽略警告
warnings.filterwarnings("ignore")


# ==========================================
# 1. 配置参数 - 结果统一管理版
# ==========================================
class Config:
    # 确保文件名与你预处理脚本生成的文件名一致
    TRAIN_FEAT_PATH = "../../data/X_train_final.csv"
    TRAIN_LABEL_PATH = "../../data/y_train_final.csv"
    TEST_FEAT_PATH = "../../data/X_test_final.csv"
    TEST_LABEL_PATH = "../../data/y_test_final.csv"

    # 统一输出文件夹
    RESULT_DIR = "../../result/lgbm_result"

    # 文件夹内的文件路径
    MODEL_PKL = os.path.join(RESULT_DIR, "lgbm_model.pkl")
    PREDS_CSV = os.path.join(RESULT_DIR, "lgbm_preds.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "lgbm_metrics_dashboard.png")
    IMPORTANCE_PNG = os.path.join(RESULT_DIR, "lgbm_feature_importance.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "lgbm_evaluation_report.txt")

    PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.005,
        "feature_pre_filter": False,
        "scale_pos_weight": 4.0,  # 针对前三名约25%-30%的占比进行平衡
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "device": "gpu",  # 如果没有GPU环境，请改为 "cpu"
        "verbose": -1,
    }
    NUM_ROUNDS = 10000
    EARLY_STOP = 200


# 创建输出目录
os.makedirs(Config.RESULT_DIR, exist_ok=True)


def log_print(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# ==========================================
# 2. 统一绘图与报告生成
# ==========================================
def generate_unified_assets(y_test, y_prob, gbm):
    log_print(f"正在生成统一评估资产至 {Config.RESULT_DIR}...")

    # --- 1. 计算指标 ---
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[best_idx]
    best_th = thresholds[best_idx]

    # --- 2. 写入文字报告 ---
    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=== LightGBM 统一评估报告 ===\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"核心指标:\n")
        f.write(f" - AUC (ROC) : {auc:.6f}\n")
        f.write(f" - Log-Loss  : {loss:.6f}\n")
        f.write(f" - Max F1    : {max_f1:.6f}\n")
        f.write(f" - 最佳阈值  : {best_th:.4f}\n")

    # --- 3. 绘制评估看板 (ROC + PR) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # (A) ROC 曲线与局部放大
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f"LGBM (AUC = {auc:.4f})", color="#1f77b4", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)

    try:
        axins = inset_axes(ax1, width="40%", height="40%", loc="lower right", borderpad=3)
        axins.plot(fpr, tpr, color="#1f77b4", lw=2)
        axins.set_xlim(0.1, 0.3)
        axins.set_ylim(0.6, 0.8)
        mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")
    except:
        pass  # 某些环境可能不支持放大镜效果

    ax1.set_title("ROC Curve")
    ax1.legend()

    # (B) PR 曲线
    ax2.plot(recalls, precisions, color="darkgreen", lw=2, label=f"F1 Max = {max_f1:.4f}")
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()

    plt.savefig(Config.PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    # --- 4. 特征重要性排行 ---
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(gbm, max_num_features=20, importance_type="gain")
    plt.title("Top 20 Features (Gain)")
    plt.savefig(Config.IMPORTANCE_PNG, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# 3. 主程序
# ==========================================
def main():
    log_print("正在加载数据...")

    # 加载训练集并剔除可能的最后两列（如果是训练集也有的话）
    df_train_x = pd.read_csv(Config.TRAIN_FEAT_PATH)
    if "raw_win_odds" in df_train_x.columns:
        X_train = df_train_x.iloc[:, :-2]
        log_print(f"训练集检测到非特征列，已剔除。剩余特征数: {X_train.shape[1]}")
    else:
        X_train = df_train_x

    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).values.ravel()

    # 加载测试集并【严格剔除】最后两列 (raw_win_odds, actual_rank)
    df_test_x = pd.read_csv(Config.TEST_FEAT_PATH)

    # 记录原始结算信息，防止后续丢失
    # 虽然这里不训练，但为了保证后面 predict 时 X_test 维度正确
    X_test = df_test_x.iloc[:, :-2]
    log_print(f"测试集已剔除最后两列 (raw_win_odds, actual_rank)。剩余特征数: {X_test.shape[1]}")

    y_test = pd.read_csv(Config.TEST_LABEL_PATH).values.ravel()

    # 构建 Dataset
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_test, y_test, reference=lgb_train)

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=Config.EARLY_STOP),
    ]

    log_print("开始训练...")
    gbm = lgb.train(
        Config.PARAMS,
        lgb_train,
        num_boost_round=Config.NUM_ROUNDS,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # 1. 导出预测结果
    # 注意：这里使用剔除后的 X_test 进行预测
    y_prob = gbm.predict(X_test)
    pd.DataFrame({"prob": y_prob, "actual_label": y_test}).to_csv(Config.PREDS_CSV, index=False)

    # 2. 生成所有评估图表与指标文件
    generate_unified_assets(y_test, y_prob, gbm)

    # 3. 序列化保存模型
    # 此时模型内部保存的 feature_names_ 已经不包含最后两列
    with open(Config.MODEL_PKL, "wb") as f:
        pickle.dump(gbm, f)

    log_print(f"✅ 任务完成！模型特征数: {len(gbm.feature_name())}")
    log_print(f"✅ 所有结果已保存至: {Config.RESULT_DIR}")


if __name__ == "__main__":
    main()
