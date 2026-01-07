"""
Stacking模型验证脚本
用于评估Stacking模型在验证集上的预测表现
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)


class Config:
    PREDS_PATH = "../result/stacking_tuning_result/stacking_final_preds.csv"
    LABEL_PATH = "../data/y_test_final.csv"
    OUTPUT_DIR = "../result/stacking_tuning_result"
    REPORT_PATH = os.path.join(OUTPUT_DIR, "validation_report.txt")
    FIGURE_PATH = os.path.join(OUTPUT_DIR, "stacking_validation_dashboard.png")


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """计算多种评估指标"""
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "AUC-ROC": roc_auc_score(y_true, y_prob),
        "LogLoss": log_loss(y_true, y_prob),
        "Brier Score": brier_score_loss(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }

    return metrics, y_pred


def find_optimal_threshold(y_true, y_prob):
    """寻找最优阈值（基于F1-Score）"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    return optimal_threshold, f1_scores[optimal_idx]


def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title("Confusion Matrix", fontsize=14)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_validation_dashboard(y_true, y_prob, optimal_threshold, save_path):
    """生成验证仪表板"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    axes[0, 0].plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {auc_score:.4f})")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0, 0].fill_between(fpr, tpr, alpha=0.2)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve", fontsize=14)
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(alpha=0.3)

    # 2. PR曲线
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[0, 1].plot(rec, prec, "g-", lw=2)
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve", fontsize=14)
    axes[0, 1].grid(alpha=0.3)

    # 3. 概率分布
    axes[0, 2].hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Negative (0)", color="blue")
    axes[0, 2].hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Positive (1)", color="red")
    axes[0, 2].axvline(optimal_threshold, color="black", linestyle="--", label=f"Threshold = {optimal_threshold:.3f}")
    axes[0, 2].set_xlabel("Predicted Probability")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title("Probability Distribution", fontsize=14)
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # 4. 混淆矩阵热力图
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_optimal)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    axes[1, 0].set_title(f"Confusion Matrix (threshold={optimal_threshold:.3f})", fontsize=14)
    axes[1, 0].set_ylabel("Actual Label")
    axes[1, 0].set_xlabel("Predicted Label")

    # 5. F1 vs Threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    axes[1, 1].plot(thresholds, f1_scores[:-1], "purple", lw=2)
    axes[1, 1].axvline(optimal_threshold, color="black", linestyle="--", label=f"Optimal = {optimal_threshold:.3f}")
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_ylabel("F1-Score")
    axes[1, 1].set_title("F1-Score vs Threshold", fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # 6. 校准曲线（简化的可靠性图）
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_prob, bins) - 1
    calibrated_probs = []
    actual_rates = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            calibrated_probs.append((bins[i] + bins[i + 1]) / 2)
            actual_rates.append(y_true[mask].mean())
    axes[1, 2].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[1, 2].plot(calibrated_probs, actual_rates, "ro-", lw=2, label="Model calibration")
    axes[1, 2].set_xlabel("Mean Predicted Probability")
    axes[1, 2].set_ylabel("Fraction of Positives")
    axes[1, 2].set_title("Calibration Curve", fontsize=14)
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(y_true, y_prob, metrics, optimal_threshold, y_pred, report_path):
    """生成文本报告"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("           Stacking 模型验证报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 60 + "\n")
        f.write("1. 数据概览\n")
        f.write("-" * 60 + "\n")
        f.write(f"   样本总数: {len(y_true):,}\n")
        f.write(f"   正样本数: {int(y_true.sum()):,}\n")
        f.write(f"   负样本数: {int((y_true == 0).sum()):,}\n")
        f.write(f"   正样本比例: {y_true.mean()*100:.2f}%\n\n")

        f.write("-" * 60 + "\n")
        f.write("2. 核心评估指标\n")
        f.write("-" * 60 + "\n")
        f.write(f"   {'指标':<15} {'值':>15}\n")
        f.write(f"   {'-'*15} {'-'*15}\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"   {key:<15} {value:>15.6f}\n")
            else:
                f.write(f"   {key:<15} {value:>15}\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("3. 阈值分析\n")
        f.write("-" * 60 + "\n")
        f.write(f"   默认阈值: 0.5\n")
        f.write(f"   最优阈值: {optimal_threshold:.4f} (基于F1-Score)\n")
        f.write(f"   最优F1: {metrics['F1-Score']:.6f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("4. 分类报告 (使用默认阈值0.5)\n")
        f.write("-" * 60 + "\n")
        report = classification_report(y_true, y_pred, target_names=["Negative (0)", "Positive (1)"])
        f.write(report + "\n\n")

        f.write("-" * 60 + "\n")
        f.write("5. 模型解读\n")
        f.write("-" * 60 + "\n")
        if metrics["AUC-ROC"] >= 0.9:
            auc_interpretation = "优秀 (AUC >= 0.9)"
        elif metrics["AUC-ROC"] >= 0.8:
            auc_interpretation = "良好 (0.8 <= AUC < 0.9)"
        elif metrics["AUC-ROC"] >= 0.7:
            auc_interpretation = "一般 (0.7 <= AUC < 0.8)"
        else:
            auc_interpretation = "较差 (AUC < 0.7)"
        
        f.write(f"   AUC-ROC: {auc_interpretation}\n")
        f.write(f"   Brier Score: {metrics['Brier Score']:.6f} (越接近0越好)\n")
        f.write(f"   准确率: {metrics['Accuracy']*100:.2f}%\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("                 报告结束\n")
        f.write("=" * 60 + "\n")


def main():
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 启动Stacking模型验证...")
    
    # 检查文件是否存在
    if not os.path.exists(Config.PREDS_PATH):
        print(f"❌ 错误: 预测文件不存在 - {Config.PREDS_PATH}")
        return
    
    if not os.path.exists(Config.LABEL_PATH):
        print(f"❌ 错误: 标签文件不存在 - {Config.LABEL_PATH}")
        return
    
    # 加载数据
    print(f"[{time.strftime('%H:%M:%S')}] 📂 加载数据...")
    y_true = pd.read_csv(Config.LABEL_PATH).iloc[:, -1].values.ravel()
    y_prob = pd.read_csv(Config.PREDS_PATH)["prob"].values
    
    print(f"[{time.strftime('%H:%M:%S')}] 样本数: {len(y_true):,}, 正样本: {int(y_true.sum()):,}")
    
    # 计算指标
    print(f"[{time.strftime('%H:%M:%S')}] 📊 计算评估指标...")
    metrics, y_pred = calculate_metrics(y_true, y_prob, threshold=0.5)
    
    # 寻找最优阈值
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_true, y_prob)
    
    # 使用最优阈值重新计算指标
    metrics_optimal, y_pred_optimal = calculate_metrics(y_true, y_prob, threshold=optimal_threshold)
    
    # 生成可视化
    print(f"[{time.strftime('%H:%M:%S'})] 🎨 生成可视化图表...")
    plot_validation_dashboard(y_true, y_prob, optimal_threshold, Config.FIGURE_PATH)
    
    # 生成混淆矩阵
    cm_fig_path = os.path.join(Config.OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred_optimal, cm_fig_path)
    
    # 生成报告
    print(f"[{time.strftime('%H:%M:%S')}] 📝 生成验证报告...")
    generate_report(y_true, y_prob, metrics_optimal, optimal_threshold, y_pred_optimal, Config.REPORT_PATH)
    
    # 打印摘要
    print("\n" + "=" * 50)
    print("📊 Stacking 模型验证结果摘要")
    print("=" * 50)
    print(f"   AUC-ROC:      {metrics_optimal['AUC-ROC']:.6f}")
    print(f"   LogLoss:      {metrics_optimal['LogLoss']:.6f}")
    print(f"   Brier Score:  {metrics_optimal['Brier Score']:.6f}")
    print(f"   Accuracy:     {metrics_optimal['Accuracy']*100:.2f}%")
    print(f"   Precision:    {metrics_optimal['Precision']*100:.2f}%")
    print(f"   Recall:       {metrics_optimal['Recall']*100:.2f}%")
    print(f"   F1-Score:     {metrics_optimal['F1-Score']:.6f}")
    print(f"   最优阈值:     {optimal_threshold:.4f}")
    print("=" * 50)
    
    print(f"\n✅ 验证完成!")
    print(f"   📈 可视化: {Config.FIGURE_PATH}")
    print(f"   📊 混淆矩阵: {cm_fig_path}")
    print(f"   📝 报告: {Config.REPORT_PATH}")


if __name__ == "__main__":
    main()
