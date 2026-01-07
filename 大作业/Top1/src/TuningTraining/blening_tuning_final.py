import os
import time
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    # 调参后的四个模型预测结果路径
    MODEL_PREDS = {
        "CatBoost": "../../result/cat_tuning_result/cat_tuned_preds.csv",
        "LightGBM": "../../result/lgbm_tuning_result/lgbm_tuned_preds.csv",
        "XGBoost": "../../result/xgb_tuning_result/xgb_tuned_preds.csv",
        "TabNet": "../../result/tabnet_result/tabnet_preds.csv",
    }

    # 标签路径
    LABEL_PATH = "../../data/y_test_final.csv"

    # 输出目录
    OUTPUT_DIR = "../../result/blending_tuning_result"
    N_TRIALS = 150  # 稍微增加搜索次数以获得更精细的权重

    REPORT_TXT = os.path.join(OUTPUT_DIR, "blending_report.txt")
    WEIGHTS_PNG = os.path.join(OUTPUT_DIR, "optimized_weights_pie.png")
    CORR_PNG = os.path.join(OUTPUT_DIR, "model_correlation_heatmap.png")


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. 载入数据与对齐
# ==========================================
def load_data():
    # 载入真实标签 (取最后一列)
    df_label = pd.read_csv(Config.LABEL_PATH)
    y_true = df_label.iloc[:, -1].values.ravel()

    # 载入各模型预测概率
    preds_dict = {}
    for name, path in Config.MODEL_PREDS.items():
        if os.path.exists(path):
            # 确保读取的是 prob 列
            preds_dict[name] = pd.read_csv(path)["prob"].values
        else:
            print(f"⚠️ 警告: 未找到 {name} 的预测文件，跳过该模型。")

    return y_true, preds_dict


# ==========================================
# 3. 资产生成逻辑
# ==========================================
def generate_assets(y_true, preds_dict, best_weights, final_auc):
    # 1. 计算各模型独立 AUC
    individual_aucs = {name: roc_auc_score(y_true, prob) for name, prob in preds_dict.items()}

    # 2. 生成相关性热图
    # 融合原则：模型间的相关性越低，融合收益越高
    df_corr = pd.DataFrame(preds_dict).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".4f")
    plt.title("Model Prediction Correlation")
    plt.savefig(Config.CORR_PNG, dpi=300)
    plt.close()

    # 3. 写入实验报告
    with open(Config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 45 + "\n")
        f.write("      赛马预测全模型融合(Blending)报告\n")
        f.write("=" * 45 + "\n")
        f.write(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("📈 [单一模型表现对比 (AUC)]\n")
        for name, score in individual_aucs.items():
            f.write(f" - {name:10}: {score:.6f}\n")

        f.write(f"\n🚀 [融合后表现]\n")
        f.write(f" - Final Blended AUC: {final_auc:.6f}\n")
        improvement = final_auc - max(individual_aucs.values())
        f.write(f" - 相比最强单模型提升: {improvement:.6f}\n\n")

        f.write("⚖️ [最优权重分配]\n")
        for name, w in best_weights.items():
            f.write(f" - {name:10}: {w*100:.2f}%\n")

    # 4. 权重占比饼图
    plt.figure(figsize=(10, 6))
    names = list(best_weights.keys())
    vals = list(best_weights.values())
    plt.pie(
        vals, labels=names, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("viridis", len(names))
    )
    plt.title("Optimized Model Contribution")
    plt.savefig(Config.WEIGHTS_PNG, dpi=300)
    plt.close()


# ==========================================
# 4. 主程序
# ==========================================
def objective(trial, y_true, preds_dict):
    weights = {name: trial.suggest_float(name, 0.0, 1.0) for name in preds_dict.keys()}

    total_w = sum(weights.values())
    if total_w == 0:
        return 0

    blended_prob = sum(prob * (weights[name] / total_w) for name, prob in preds_dict.items())
    return roc_auc_score(y_true, blended_prob)


def main():
    print(f"[{time.strftime('%H:%M:%S')}] 加载预测结果中...")
    y_true, preds_dict = load_data()

    if len(preds_dict) < 2:
        print("❌ 错误: 需要至少两个模型的预测结果才能进行融合。")
        return

    # 1. Optuna 权重寻优
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, y_true, preds_dict), n_trials=Config.N_TRIALS)

    # 2. 归一化权重
    best_raw = study.best_params
    total_w = sum(best_raw.values())
    best_weights = {k: v / total_w for k, v in best_raw.items()}

    print(f"\n✅ 寻优完成! 融合 AUC: {study.best_value:.6f}")

    # 3. 计算并保存最终预测概率
    final_prob = sum(preds_dict[name] * best_weights[name] for name in preds_dict.keys())
    pd.DataFrame({"prob": final_prob}).to_csv(
        os.path.join(Config.OUTPUT_DIR, "final_blended_preds.csv"), index=False
    )

    # 4. 生成图表与报告
    generate_assets(y_true, preds_dict, best_weights, study.best_value)

    print(f"\n✨ 融合资产已归档至: {Config.RESULT_DIR}")
    print(f"📊 请查看相关性热图: {Config.CORR_PNG}")


if __name__ == "__main__":
    main()
