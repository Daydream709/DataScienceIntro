import os
import time
import pickle
import warnings
import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns
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

    RESULT_DIR = "../../result/cat_tuning_result"
    N_TRIALS = 30

    BEST_PARAMS_PKL = os.path.join(RESULT_DIR, "best_params.pkl")
    MODEL_PKL = os.path.join(RESULT_DIR, "cat_tuned_model.pkl")
    PREDS_CSV = os.path.join(RESULT_DIR, "cat_tuned_preds.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "cat_tuned_dashboard.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "cat_tuning_report.txt")


os.makedirs(Config.RESULT_DIR, exist_ok=True)


# ==========================================
# 2. Optuna 目标函数
# ==========================================
def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
        "random_strength": trial.suggest_float("random_strength", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 8.0, 15.0),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "devices": "0",
        "bootstrap_type": "Bayesian",
        "verbose": False,
        "allow_writing_files": False,
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100, use_best_model=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)


# ==========================================
# 3. 资产生成 (包含 1x3 画布与 TXT 报告)
# ==========================================
def generate_assets(y_test, y_prob, model, params):
    # --- 1. 计算指标 ---
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    prec, rec, thres = precision_recall_curve(y_test, y_prob)

    # 自动计算 F1 最大时的最佳阈值
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    ix = np.argmax(f1)
    max_f1 = f1[ix]
    best_th = thres[ix] if ix < len(thres) else 0.5

    # --- 2. 提取特征重要性 ---
    imp_df = pd.DataFrame(
        {"feature": model.feature_names_, "importance": model.get_feature_importance()}
    ).sort_values(by="importance", ascending=False)

    # --- 3. 写入实验报告 ---
    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write("      CatBoost 自动调参实验报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("🏆 [最佳超参数配置]\n")
        for k, v in params.items():
            f.write(f" - {k:20}: {v}\n")

        f.write("\n📈 [核心表现指标]\n")
        f.write(f" - AUC (排序能力)   : {auc:.6f}\n")
        f.write(f" - LogLoss (确定性) : {loss:.6f}\n")
        f.write(f" - Max F1 Score    : {max_f1:.6f}\n")
        f.write(f" - Best Threshold  : {best_th:.4f}\n")

        f.write("\n📊 [特征重要性 TOP 20]\n")
        for i, row in imp_df.head(20).reset_index(drop=True).iterrows():
            f.write(f" {i+1:2}. {row['feature']:25}: {row['importance']:.4f}\n")

    # --- 4. 综合绘图 (1x3 看板) ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # 子图 1: ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f"CatBoost (AUC={auc:.4f})", color="#e74c3c", lw=3)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC Curve", fontsize=14)
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # 子图 2: PR 曲线
    axes[1].plot(rec, prec, color="#c0392b", lw=3, label=f"Max F1={max_f1:.3f}")
    axes[1].set_title("Precision-Recall Curve", fontsize=14)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 子图 3: 特征重要性条形图
    top_20 = imp_df.head(20).sort_values(by="importance", ascending=True)
    axes[2].barh(top_20["feature"], top_20["importance"], color="#e74c3c")
    axes[2].set_title("Top 20 Feature Importance", fontsize=14)
    axes[2].set_xlabel("Feature Importance Score")
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.PLOT_PNG, dpi=300)
    plt.close()


# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 正在读取数据并对齐特征...")
    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    # 1. 明确剔除非特征列
    drop_cols = ["raw_win_odds", "actual_rank", "race_id", "win_odds"]
    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_test_raw = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    # 2. 强制对齐特征
    feature_names = X_train.columns.tolist()
    X_test = X_test_raw.reindex(columns=feature_names, fill_value=0)

    # 3. 加载标签
    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    print(f"✅ 数据准备就绪 | 特征数: {len(feature_names)} | 样本数: {len(X_train)}")

    # 4. 自动调参
    print(f"🧬 启动 Optuna 寻优 (试验次数: {Config.N_TRIALS})...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=Config.N_TRIALS)

    best_p = study.best_params
    print(f"\n✅ 寻优完成! 最佳测试集 AUC: {study.best_value:.6f}")

    # 5. 终极模型训练
    print("🏗️ 正在训练终极模型 (iterations=10000)...")
    final_params = {
        "iterations": 10000,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "early_stopping_rounds": 300,
        "verbose": 100,
        **best_p,
    }

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # 6. 保存资产
    y_prob = final_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"prob": y_prob}).to_csv(Config.PREDS_CSV, index=False)

    generate_assets(y_test, y_prob, final_model, best_p)

    with open(Config.MODEL_PKL, "wb") as f:
        pickle.dump(final_model, f)
    with open(Config.BEST_PARAMS_PKL, "wb") as f:
        pickle.dump(best_p, f)

    print(f"\n✨ CatBoost 调参看板已生成: {Config.PLOT_PNG}")
    print(f"✨ 实验报告已保存至: {Config.REPORT_TXT}")


if __name__ == "__main__":
    main()
