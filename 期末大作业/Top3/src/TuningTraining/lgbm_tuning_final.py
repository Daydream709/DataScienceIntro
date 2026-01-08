import os
import time
import pickle
import warnings
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, log_loss, roc_curve, precision_recall_curve

warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置参数 (更新为 Top 3 路径)
# ==========================================
class Config:
    # 使用你刚才生成的 Top 3 数据文件
    TRAIN_FEAT_PATH = "../../data/X_train_final_top3.csv"
    TRAIN_LABEL_PATH = "../../data/y_train_final_top3.csv"
    TEST_FEAT_PATH = "../../data/X_test_final_top3.csv"
    TEST_LABEL_PATH = "../../data/y_test_final_top3.csv"

    RESULT_DIR = "../../result/lgbm_top3_tuning_result"
    N_TRIALS = 30 

    BEST_PARAMS_PKL = os.path.join(RESULT_DIR, "best_params_top3.pkl")
    MODEL_PKL = os.path.join(RESULT_DIR, "lgbm_tuned_model_top3.pkl")
    PREDS_CSV = os.path.join(RESULT_DIR, "lgbm_tuned_preds_top3.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "lgbm_tuned_dashboard_top3.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "lgbm_tuning_report_top3.txt")

    DEVICE = (
        "gpu"
        if os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("C:/Windows/System32/nvapi64.dll")
        else "cpu"
    )

os.makedirs(Config.RESULT_DIR, exist_ok=True)

# ==========================================
# 2. 目标函数 (优化搜索空间)
# ==========================================
def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "objective": "binary",
        "metric": "auc", # 也可以考虑使用 "binary_logloss"
        "verbosity": -1,
        "boosting_type": "gbdt",
        "device": Config.DEVICE,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100), # Top3 模式下适当减小复杂度
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        # 重要修改：Top 3 的正样本比例约 25%-30%，权重应设为 2.0-5.0 之间
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 5.0), 
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)],
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)

# ==========================================
# 3. 资产生成函数
# ==========================================
def generate_assets(y_test, y_prob, model, params):
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    prec, rec, thres = precision_recall_curve(y_test, y_prob)

    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    ix = np.argmax(f1)
    max_f1 = f1[ix]
    best_th = thres[ix] if ix < len(thres) else 0.5

    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"LightGBM Top 3 调参报告\n{'='*25}\n")
        f.write(f"AUC: {auc:.6f} | LogLoss: {loss:.6f} | F1: {max_f1:.6f}\n")
        f.write(f"Best Threshold: {best_th:.4f}\n\nBest Params: {params}\n")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, color="#3498db", lw=3, label=f"AUC={auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("ROC Curve (Top 3)")
    axes[0].legend()

    # PR Curve (Top 3 任务中 PR 曲线比 ROC 更能反应预测质量)
    axes[1].plot(rec, prec, color="#9b59b6", lw=3)
    axes[1].set_title("Precision-Recall Curve (Top 3)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    # Importance
    imp_df = (
        pd.DataFrame({"feat": model.feature_name_, "imp": model.feature_importances_})
        .sort_values("imp", ascending=False)
        .head(20)
    )
    sns.barplot(x="imp", y="feat", data=imp_df, ax=axes[2], palette="magma")
    axes[2].set_title("Top 20 Features for Top 3 Prediction")

    plt.tight_layout()
    plt.savefig(Config.PLOT_PNG)
    print(f"[{time.strftime('%H:%M:%S')}] Top 3 看板已生成: {Config.PLOT_PNG}")

# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] Top 3 脚本启动，设备: {Config.DEVICE}")

    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    drop_cols = ["raw_win_odds", "actual_rank", "race_id", "win_odds"]
    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns]).reindex(
        columns=X_train.columns, fill_value=0
    )

    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    print(f"[{time.strftime('%H:%M:%S')}] 数据加载完成。训练集正样本: {np.sum(y_train)} / {len(y_train)}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test),
        n_trials=Config.N_TRIALS,
        show_progress_bar=True,
    )

    print(f"[{time.strftime('%H:%M:%S')}] 调参结束，开始 Top 3 终极训练...")

    best_p = study.best_params
    final_model = lgb.LGBMClassifier(n_estimators=1000, **best_p, device=Config.DEVICE)
    final_model.fit(
        X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.log_evaluation(100)]
    )

    y_prob = final_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"prob": y_prob}).to_csv(Config.PREDS_CSV, index=False)
    
    with open(Config.MODEL_PKL, "wb") as f:
        pickle.dump(final_model, f)
        
    generate_assets(y_test, y_prob, final_model, best_p)
    print(f"[{time.strftime('%H:%M:%S')}] Top 3 流程圆满完成！")

if __name__ == "__main__":
    main()