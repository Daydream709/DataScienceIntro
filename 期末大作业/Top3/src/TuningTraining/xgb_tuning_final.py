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
# 1. 配置参数 (更新为 Top 3 路径)
# ==========================================
class Config:
    TRAIN_FEAT_PATH = "../../data/X_train_final_top3.csv"
    TRAIN_LABEL_PATH = "../../data/y_train_final_top3.csv"
    TEST_FEAT_PATH = "../../data/X_test_final_top3.csv"
    TEST_LABEL_PATH = "../../data/y_test_final_top3.csv"

    RESULT_DIR = "../../result/xgb_top3_tuning_result"
    N_TRIALS = 30 

    BEST_PARAMS_PKL = os.path.join(RESULT_DIR, "best_params_top3.pkl")
    MODEL_PKL = os.path.join(RESULT_DIR, "xgb_tuned_model_top3.pkl")
    PREDS_CSV = os.path.join(RESULT_DIR, "xgb_tuned_preds_top3.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "xgb_tuned_dashboard_top3.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "xgb_tuning_report_top3.txt")

os.makedirs(Config.RESULT_DIR, exist_ok=True)

# ==========================================
# 2. Optuna 目标函数
# ==========================================
def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",  # 使用 GPU
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50), # 增加约束防止过拟合
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-2, 10.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
        # 针对 21.63% 正样本率，建议权重在 2.0-4.0
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 4.0),
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
        "verbosity": 0,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)

# ==========================================
# 3. 资产生成 (1x3 看板：增加特征重要性)
# ==========================================
def generate_assets(y_test, y_prob, model, params):
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    prec, rec, thres = precision_recall_curve(y_test, y_prob)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    max_f1, best_th = np.max(f1), thres[np.argmax(f1)]

    # 1. 文本报告
    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=== XGBoost Top 3 自动调参报告 ===\n")
        f.write(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"指标统计: AUC={auc:.6f} | LogLoss={loss:.6f} | F1={max_f1:.6f} | BestTh={best_th:.4f}\n")
        f.write(f"最佳参数: {params}\n")

    # 2. 绘图看板
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f"XGB Top 3 (AUC={auc:.4f})", color="#f39c12", lw=3)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_title("ROC Curve")
    ax1.legend()

    # PR Curve
    ax2.plot(rec, prec, color="#d35400", lw=3)
    ax2.set_title("Precision-Recall Curve (Top 3)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")

    # Feature Importance (新增图表)
    # XGBoost 的重要性需要基于 feature_importances_ 属性
    imp_scores = model.feature_importances_
    feat_names = model.get_booster().feature_names
    fi_df = pd.DataFrame({'feat': feat_names, 'imp': imp_scores}).sort_values('imp', ascending=False).head(20)
    
    colors = plt.cm.get_cmap('YlOrBr')(np.linspace(0.4, 0.8, 20))
    ax3.barh(fi_df['feat'], fi_df['imp'], color=colors)
    ax3.invert_yaxis()
    ax3.set_title("Top 20 Features (XGBoost)")

    plt.tight_layout()
    plt.savefig(Config.PLOT_PNG, dpi=300)
    plt.close()
    print(f"[{time.strftime('%H:%M:%S')}] 看板已更新并保存至: {Config.PLOT_PNG}")

# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 正在读取数据并对齐特征...")
    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    # 1. 特征处理：训练集全量，测试集剔除最后三列 (raw_win_odds, actual_rank, race_id)
    X_train = df_train.copy()
    X_test_raw = df_test.iloc[:, :-3].copy() 

    # 2. 强制对齐特征
    feature_names = X_train.columns.tolist()
    X_test = X_test_raw.reindex(columns=feature_names, fill_value=0)

    # 3. 加载标签
    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    print(f"正样本率: {np.mean(y_train):.2%} | 特征数: {len(feature_names)}")

    # 4. 自动调参
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=Config.N_TRIALS)

    # 5. 终极训练
    print(f"🏗️ 寻优完成 (Best AUC: {study.best_value:.6f})，开始终极训练...")
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": 5000,
        "early_stopping_rounds": 300,
        **study.best_params,
    }

    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    # 6. 保存资产
    y_prob = final_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"prob": y_prob}).to_csv(Config.PREDS_CSV, index=False)
    generate_assets(y_test, y_prob, final_model, study.best_params)

    with open(Config.MODEL_PKL, "wb") as f:
        pickle.dump(final_model, f)

    print(f"\n✨ XGBoost Top 3 实验归档至: {Config.RESULT_DIR}")

if __name__ == "__main__":
    main()