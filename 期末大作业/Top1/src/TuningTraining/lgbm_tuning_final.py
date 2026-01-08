import os
import time
import pickle
import warnings
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns  # 增加美化
from sklearn.metrics import roc_auc_score, log_loss, roc_curve, precision_recall_curve

warnings.filterwarnings("ignore")


# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    TRAIN_FEAT_PATH = "../../data/X_train_final.csv"
    TRAIN_LABEL_PATH = "../../data/y_train_final.csv"
    TEST_FEAT_PATH = "../../data/X_test_final.csv"
    TEST_LABEL_PATH = "../../data/y_test_final.csv"

    RESULT_DIR = "../../result/lgbm_tuning_result"
    N_TRIALS = 30  # 建议先设为 5 测试一下流程

    # 路径资产
    BEST_PARAMS_PKL = os.path.join(RESULT_DIR, "best_params.pkl")
    MODEL_PKL = os.path.join(RESULT_DIR, "lgbm_tuned_model.pkl")
    PREDS_CSV = os.path.join(RESULT_DIR, "lgbm_tuned_preds.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "lgbm_tuned_dashboard.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "lgbm_tuning_report.txt")

    # 自动检测 GPU
    DEVICE = (
        "gpu"
        if os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("C:/Windows/System32/nvapi64.dll")
        else "cpu"
    )


os.makedirs(Config.RESULT_DIR, exist_ok=True)


# ==========================================
# 2. 目标函数
# ==========================================
def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "device": Config.DEVICE,  # 自动适配
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 15.0),
    }

    model = lgb.LGBMClassifier(**params)

    # 调参阶段减少打印，但保留逻辑
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)],
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)


# ==========================================
# 3. 资产生成函数 (1x3 看板)
# ==========================================
def generate_assets(y_test, y_prob, model, params):
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    prec, rec, thres = precision_recall_curve(y_test, y_prob)

    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    ix = np.argmax(f1)
    max_f1 = f1[ix]
    best_th = thres[ix] if ix < len(thres) else 0.5

    # 1. 文本报告
    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"LightGBM 调参报告\n{'='*20}\n")
        f.write(f"AUC: {auc:.6f} | LogLoss: {loss:.6f} | F1: {max_f1:.6f}\n")
        f.write(f"Best Threshold: {best_th:.4f}\n\nBest Params: {params}\n")

    # 2. 绘图看板
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, color="#2ecc71", lw=3, label=f"AUC={auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # PR
    axes[1].plot(rec, prec, color="#e67e22", lw=3)
    axes[1].set_title("Precision-Recall Curve")

    # Importance
    imp_df = (
        pd.DataFrame({"feat": model.feature_name_, "imp": model.feature_importances_})
        .sort_values("imp", ascending=False)
        .head(20)
    )
    sns.barplot(x="imp", y="feat", data=imp_df, ax=axes[2], palette="viridis")
    axes[2].set_title("Top 20 Features")

    plt.tight_layout()
    plt.savefig(Config.PLOT_PNG)
    print(f"[{time.strftime('%H:%M:%S')}] 看板图片已生成: {Config.PLOT_PNG}")


# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 脚本已启动，使用设备: {Config.DEVICE}")

    # 读取数据
    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    drop_cols = ["raw_win_odds", "actual_rank", "race_id", "win_odds"]
    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns]).reindex(
        columns=X_train.columns, fill_value=0
    )

    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    print(f"[{time.strftime('%H:%M:%S')}] 数据对齐完成，开始调参...")

    # 调参
    study = optuna.create_study(direction="maximize")
    # 核心：添加 show_progress_bar
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test),
        n_trials=Config.N_TRIALS,
        show_progress_bar=True,
    )

    print(f"[{time.strftime('%H:%M:%S')}] 调参结束，开始终极训练...")

    # 最终训练
    best_p = study.best_params
    final_model = lgb.LGBMClassifier(n_estimators=1000, **best_p, device=Config.DEVICE)
    final_model.fit(
        X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.log_evaluation(100)]
    )  # 终极训练每100步打印一次进度

    # 保存
    y_prob = final_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"prob": y_prob}).to_csv(Config.PREDS_CSV, index=False)
    generate_assets(y_test, y_prob, final_model, best_p)

    print(f"[{time.strftime('%H:%M:%S')}] 流程圆满完成！")


if __name__ == "__main__":
    main()
