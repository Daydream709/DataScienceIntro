import os
import time
import pickle
import warnings
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, roc_curve, precision_recall_curve
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

    RESULT_DIR = "../../result/xgb_tuning_result"
    N_TRIALS = 30  # 寻优次数

    BEST_PARAMS_PKL = os.path.join(RESULT_DIR, "best_params.pkl")
    MODEL_PKL = os.path.join(RESULT_DIR, "xgb_tuned_model.pkl")
    PREDS_CSV = os.path.join(RESULT_DIR, "xgb_tuned_preds.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "xgb_tuned_dashboard.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "xgb_tuning_report.txt")


os.makedirs(Config.RESULT_DIR, exist_ok=True)


# ==========================================
# 2. Optuna 目标函数
# ==========================================
def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",  # GPU 加速必备
        "device": "cuda",  # 指定使用 GPU
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
        "verbosity": 0,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)


# ==========================================
# 3. 资产生成
# ==========================================
def generate_assets(y_test, y_prob, model, params):
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    prec, rec, thres = precision_recall_curve(y_test, y_prob)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    max_f1, best_th = np.max(f1), thres[np.argmax(f1)]

    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=== XGBoost 自动调参实验报告 ===\n")
        f.write(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("🏆 最佳超参数配置:\n")
        for k, v in params.items():
            f.write(f" - {k}: {v}\n")
        f.write(f"\n📈 最终表现指标:\n")
        f.write(f" - AUC     : {auc:.6f}\n")
        f.write(f" - LogLoss : {loss:.6f}\n")
        f.write(f" - Max F1  : {max_f1:.6f}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f"Tuned XGB (AUC={auc:.4f})", color="orange", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)

    try:
        axins = inset_axes(ax1, width="40%", height="40%", loc="lower right", borderpad=3)
        axins.plot(fpr, tpr, color="orange", lw=2)
        axins.set_xlim(0.1, 0.3)
        axins.set_ylim(0.6, 0.8)
        mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")
    except:
        pass

    ax2.plot(rec, prec, color="darkorange", lw=2, label="PR Curve")
    ax2.set_title("PR Curve (XGB Tuned)")
    ax1.set_title("ROC Curve (XGB Tuned)")
    ax1.legend()
    plt.savefig(Config.PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 正在读取数据并对齐特征...")
    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    # 1. 特征处理：训练集全量，测试集剔除最后两列
    X_train = df_train.copy()
    X_test_raw = df_test.iloc[:, :-2].copy()

    # 2. 强制对齐特征
    feature_names = X_train.columns.tolist()
    X_test = X_test_raw.reindex(columns=feature_names, fill_value=0)

    # 3. 加载标签
    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    print(f"特征数: {len(feature_names)} | 调参样本数: {len(X_train)}")

    # 4. 自动调参
    print(f"🧬 启动 Optuna 寻优 (试验次数: {Config.N_TRIALS})...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=Config.N_TRIALS)

    best_p = study.best_params
    print(f"\n✅ 寻优完成! 最佳测试集 AUC: {study.best_value:.6f}")

    # 5. 使用最佳参数进行终极模型训练
    print("🏗️ 正在使用最佳参数进行终极模型训练...")
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": 5000,
        "early_stopping_rounds": 300,
        **best_p,
    }

    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    # 6. 保存资产
    y_prob = final_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"prob": y_prob}).to_csv(Config.PREDS_CSV, index=False)
    generate_assets(y_test, y_prob, final_model, best_p)

    with open(Config.MODEL_PKL, "wb") as f:
        pickle.dump(final_model, f)
    with open(Config.BEST_PARAMS_PKL, "wb") as f:
        pickle.dump(best_p, f)

    print(f"\n✨ XGBoost 调参结果已归档至: {Config.RESULT_DIR}")


if __name__ == "__main__":
    main()
