import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 路径配置
# ==========================================
class Config:
    # 填入你想要分析的预测结果路径
    PREDS_PATH = "../result/cat_tuning_result/cat_tuned_preds.csv"
    TEST_DATA_PATH = "../data/X_test_final.csv"

def analyze_pure_accuracy():
    # 1. 加载数据
    if not os.path.exists(Config.PREDS_PATH):
        print("❌ 找不到预测文件")
        return

    df_preds = pd.read_csv(Config.PREDS_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # 2. 整合关键列
    # 我们只需要：场次ID、实际排名、模型预测概率
    analysis_df = pd.DataFrame({
        "race_id": df_test["race_id"].values,
        "actual_rank": df_test["actual_rank"].values,
        "model_prob": df_preds["prob"].values
    })

    # 3. 核心逻辑：找出模型在每场比赛中预测概率最高的马
    # 按 race_id 分组，并提取 model_prob 最大的那一行
    model_favorites = analysis_df.loc[analysis_df.groupby("race_id")["model_prob"].idxmax()].copy()

    # 4. 计算指标
    total_races = len(model_favorites)
    # 命中数：模型看好的马实际排名是 1
    hit_count = len(model_favorites[model_favorites["actual_rank"] == 1])
    # 计算准确率
    accuracy = (hit_count / total_races) * 100 if total_races > 0 else 0

    # 5. 进阶：计算前三名命中率 (Top-3 Accuracy)
    # 只要模型预测的前三个概率中包含了真实冠军，就算中
    analysis_df['model_rank'] = analysis_df.groupby('race_id')['model_prob'].rank(ascending=False)
    top3_hits = analysis_df[(analysis_df['model_rank'] <= 3) & (analysis_df['actual_rank'] == 1)]
    top3_accuracy = (len(top3_hits) / total_races) * 100 if total_races > 0 else 0

    # ==========================================
    # 输出分析结果
    # ==========================================
    print("\n" + "═"*40)
    print(f"📊 模型真实预测力分析报告")
    print("═"*40)
    print(f"🏁 总分析场次数:      {total_races}")
    print(f"🥇 模型 Top-1 命中数: {hit_count}")
    print(f"🎯 真实准确率 (Win):  {accuracy:.2f}%")
    print(f"🥉 模型 Top-3 命中率: {top3_accuracy:.2f}%")
    print("═"*40)

if __name__ == "__main__":
    analyze_pure_accuracy()