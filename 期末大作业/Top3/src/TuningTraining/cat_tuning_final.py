import os
import time
import pickle
import warnings
import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, roc_curve, precision_recall_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置参数 (更新为 Top 3 路径)
# ==========================================
class Config:
    # 使用 Top 3 专用数据
    TRAIN_FEAT_PATH = "../../data/X_train_final_top3.csv"
    TRAIN_LABEL_PATH = "../../data/y_train_final_top3.csv"
    TEST_FEAT_PATH = "../../data/X_test_final_top3.csv"
    TEST_LABEL_PATH = "../../data/y_test_final_top3.csv"

    RESULT_DIR = "../../result/cat_top3_tuning_result"
    N_TRIALS = 30

    BEST_PARAMS_PKL = os.path.join(RESULT_DIR, "best_params_top3.pkl")
    MODEL_PKL = os.path.join(RESULT_DIR, "cat_tuned_model_top3.pkl")
    PREDS_CSV = os.path.join(RESULT_DIR, "cat_tuned_preds_top3.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "cat_tuned_dashboard_top3.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "cat_tuning_report_top3.txt")

os.makedirs(Config.RESULT_DIR, exist_ok=True)

# ==========================================
# 2. Optuna 目标函数 (针对 Top 3 优化)
# ==========================================
def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "depth": trial.suggest_int("depth", 5, 8), # Top 3 任务稍浅的树往往泛化更好
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        "random_strength": trial.suggest_float("random_strength", 0.8, 3.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        # 核心：正样本比例 21% 时，权重建议在 2.5 - 4.5 之间
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.5, 4.5),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "devices": "0",
        "bootstrap_type": "Bayesian",
        "verbose": False,
        "allow_writing_files": False,
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100)

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

    # 1. 写入详细报告
    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=== CatBoost Top 3 自动调参报告 ===\n")
        f.write(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("🏆 最佳超参数配置:\n")
        for k, v in params.items(): f.write(f" - {k}: {v}\n")
        f.write(f"\n📈 性能指标:\n")
        f.write(f" - AUC     : {auc:.6f}\n")
        f.write(f" - LogLoss : {loss:.6f}\n")
        f.write(f" - Max F1  : {max_f1:.6f}\n")
        f.write(f" - Best Th : {best_th:.4f}\n")

    # 2. 创建 1x3 的看板
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    
    # --- 图 1: ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f"CatBoost Top 3 (AUC={auc:.4f})", color="#e74c3c", lw=3)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_title("ROC Curve (Top 3 Tasks)")
    ax1.legend()
    
    # --- 图 2: PR Curve ---
    ax2.plot(rec, prec, color="#c0392b", lw=3)
    ax2.set_title("Precision-Recall (Top 3 Quality)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    
    # --- 图 3: Feature Importance (新增) ---
    # 获取特征重要性
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_
    
    # 整理数据并排序
    fi_df = pd.DataFrame({'feat': feature_names, 'imp': feature_importance})
    fi_df = fi_df.sort_values(by='imp', ascending=False).head(20) # 只取前20个
    
    # 绘制条形图
    colors = plt.cm.get_cmap('YlOrRd_r')(np.linspace(0.2, 0.7, 20)) # 使用红橙色系
    ax3.barh(fi_df['feat'], fi_df['imp'], color=colors)
    ax3.invert_yaxis() # 让最重要的排在最上面
    ax3.set_title("Top 20 Features (CatBoost)")
    ax3.set_xlabel("Feature Importance Score")

    plt.tight_layout()
    plt.savefig(Config.PLOT_PNG, dpi=300)
    plt.close()
    print(f"[{time.strftime('%H:%M:%S')}] 三合一可视化看板已生成: {Config.PLOT_PNG}")

# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 启动 Top 3 CatBoost 实验...")
    
    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    # 特征对齐逻辑
    # 训练集是全量特征，测试集需去掉末尾 3 列 (raw_win_odds, actual_rank, race_id)
    X_train = df_train.copy()
    X_test_raw = df_test.iloc[:, :-3].copy() 

    feature_names = X_train.columns.tolist()
    X_test = X_test_raw.reindex(columns=feature_names, fill_value=0)

    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    print(f"训练集正样本率: {np.mean(y_train):.2%}")

    # Optuna 寻优
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=Config.N_TRIALS)

    print(f"\n✅ 寻优完成! 最佳 AUC: {study.best_value:.6f}")

    # 终极训练
    final_params = {
        "iterations": 8000, # Top 3 样本多，迭代次数可适当减少
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "early_stopping_rounds": 200,
        "verbose": 100,
        **study.best_params,
    }

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # 保存
    y_prob = final_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"prob": y_prob}).to_csv(Config.PREDS_CSV, index=False)
    generate_assets(y_test, y_prob, final_model, study.best_params)

    with open(Config.MODEL_PKL, "wb") as f: pickle.dump(final_model, f)
    print(f"\n✨ Top 3 实验归档至: {Config.RESULT_DIR}")

if __name__ == "__main__":
    main()