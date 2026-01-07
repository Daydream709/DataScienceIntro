import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, log_loss, roc_curve, precision_recall_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

warnings.filterwarnings("ignore")


class Config:
    # 确保路径指向你最新的生成文件
    TRAIN_FEAT_PATH = "../../data/X_train_final.csv"
    TRAIN_LABEL_PATH = "../../data/y_train_final.csv"
    TEST_FEAT_PATH = "../../data/X_test_final.csv"
    TEST_LABEL_PATH = "../../data/y_test_final.csv"

    RESULT_DIR = "../../result/cat_result"
    MODEL_PKL = os.path.join(RESULT_DIR, "cat_model.pkl")
    PREDS_CSV = os.path.join(RESULT_DIR, "cat_preds.csv")
    PLOT_PNG = os.path.join(RESULT_DIR, "cat_metrics_dashboard.png")
    IMPORTANCE_PNG = os.path.join(RESULT_DIR, "cat_feature_importance.png")
    REPORT_TXT = os.path.join(RESULT_DIR, "cat_evaluation_report.txt")

    PARAMS = {
        "iterations": 10000,
        "learning_rate": 0.005,
        "depth": 8,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "task_type": "GPU",
        "devices": "0",
        "scale_pos_weight": 4.0,
        "early_stopping_rounds": 200,
        "verbose": 100,
    }


os.makedirs(Config.RESULT_DIR, exist_ok=True)


def generate_assets(y_test, y_prob, model):
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    prec, rec, thres = precision_recall_curve(y_test, y_prob)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    max_f1, best_th = np.max(f1), thres[np.argmax(f1)]

    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(
            f"=== CatBoost 统一评估报告 ===\nAUC: {auc:.6f}\nLogLoss: {loss:.6f}\nMax F1: {max_f1:.6f}\nBest Thresh: {best_th:.4f}"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f"Cat (AUC = {auc:.4f})", color="#2ca02c", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    try:
        axins = inset_axes(ax1, width="40%", height="40%", loc="lower right", borderpad=3)
        axins.plot(fpr, tpr, color="#2ca02c", lw=2)
        axins.set_xlim(0.1, 0.3)
        axins.set_ylim(0.6, 0.8)
        mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")
    except:
        pass
    ax2.plot(rec, prec, color="green", lw=2, label=f"F1 Max = {max_f1:.4f}")
    plt.savefig(Config.PLOT_PNG, dpi=300)
    plt.close()

    feat_imp = model.get_feature_importance()
    feat_names = model.feature_names_
    idx = np.argsort(feat_imp)[-20:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), feat_imp[idx], align="center")
    plt.yticks(range(20), [feat_names[i] for i in idx])
    plt.title("Top 20 Features (CatBoost)")
    plt.savefig(Config.IMPORTANCE_PNG, dpi=300)
    plt.close()


def main():
    print(f"[{time.strftime('%H:%M:%S')}] 正在读取数据...")
    df_train = pd.read_csv(Config.TRAIN_FEAT_PATH)
    df_test = pd.read_csv(Config.TEST_FEAT_PATH)

    # 1. 核心修复逻辑：
    # 训练集：直接全量作为特征 (因为里面没有 odds 和 rank)
    X_train = df_train.copy()

    # 测试集：剔除最后两列 (raw_win_odds 和 actual_rank)
    X_test_temp = df_test.iloc[:, :-2].copy()

    # 2. 强制对齐特征 (防止独热编码导致的列数微差)
    # 以训练集的列名为基准，对齐测试集
    feature_names = X_train.columns.tolist()
    X_test = X_test_temp.reindex(columns=feature_names, fill_value=0)

    # 3. 加载标签 (只取最后一列)
    y_train = pd.read_csv(Config.TRAIN_LABEL_PATH).iloc[:, -1].values.ravel()
    y_test = pd.read_csv(Config.TEST_LABEL_PATH).iloc[:, -1].values.ravel()

    print(f"[{time.strftime('%H:%M:%S')}] 数据对齐完成：")
    print(f" - 训练集特征数: {X_train.shape[1]}")
    print(f" - 测试集特征数: {X_test.shape[1]}")

    # 4. 构建训练 Pool
    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    test_pool = Pool(X_test, y_test, feature_names=feature_names)

    # 5. 训练模型
    model = CatBoostClassifier(**Config.PARAMS)
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    # 6. 预测与导出结果 (使用对齐后的 X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"prob": y_prob}).to_csv(Config.PREDS_CSV, index=False)

    generate_assets(y_test, y_prob, model)
    with open(Config.MODEL_PKL, "wb") as f:
        pickle.dump(model, f)

    print(f"[{time.strftime('%H:%M:%S')}] 任务成功完成！")


if __name__ == "__main__":
    main()
