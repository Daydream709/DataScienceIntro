import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 配置参数 (指向你 Stacking 后的结果)
# ==========================================
class Config:
    # 终极融合后的概率文件
    PRED_PATH = "../result/stacking_top3_result/stacking_final_preds_top3.csv"
    # 测试集原始特征和标签 (包含 race_id 和 actual_rank)
    TEST_DATA_PATH = "../data/X_test_final_top3.csv" 
    LABEL_PATH = "../data/y_test_final_top3.csv"
    
    OUTPUT_DIR = "../result/hit_rate_analysis"

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 数据准备与对齐
# ==========================================
def analyze_accuracy():
    print("📊 正在进行 Top-K 命中率深度分析...")
    
    # 读取概率、特征和标签
    df_prob = pd.read_csv(Config.PRED_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)
    df_label = pd.read_csv(Config.LABEL_PATH)
    
    # 合并成一个分析大表
    # 假设 df_test 的最后三列是 ['raw_win_odds', 'actual_rank', 'race_id']
    analysis_df = pd.DataFrame({
        'race_id': df_test['race_id'],
        'actual_rank': df_test['actual_rank'],
        'pred_prob': df_prob['prob'],
        'is_top3': df_label.iloc[:, -1] # 真实的前三标签
    })
    
    # ==========================================
    # 3. 按场次计算 Top-N 表现
    # ==========================================
    results = []
    
    # 按场次分组
    grouped = analysis_df.groupby('race_id')
    
    for race_id, group in grouped:
        # 对该场比赛的马匹按预测概率降序排列
        group = group.sort_values(by='pred_prob', ascending=False).reset_index(drop=True)
        
        # 1. 检查模型预测的第一名是否真的进了前三/拿了第一
        top1_actual_rank = group.loc[0, 'actual_rank']
        hit_win = 1 if top1_actual_rank == 1 else 0
        hit_place = 1 if top1_actual_rank <= 3 else 0
        
        # 2. 检查模型预测的前三名 (Top 3) 中有多少真的跑进前三
        top3_preds = group.head(3)
        hits_in_top3 = top3_preds[top3_preds['actual_rank'] <= 3].shape[0]
        
        results.append({
            'race_id': race_id,
            'top1_win_hit': hit_win,
            'top1_place_hit': hit_place,
            'top3_hits_count': hits_in_top3  # 取值范围 0, 1, 2, 3
        })
        
    res_df = pd.DataFrame(results)
    
    # ==========================================
    # 4. 统计汇总
    # ==========================================
    total_races = len(res_df)
    avg_top1_win = res_df['top1_win_hit'].mean()
    avg_top1_place = res_df['top1_place_hit'].mean()
    
    # 计算 Top-3 的分布情况
    hit_counts = res_df['top3_hits_count'].value_counts(normalize=True).sort_index()
    
    print(f"\n--- 最终准确率报告 (总比赛场次: {total_races}) ---")
    print(f"✅ 模型首选马夺冠率 (Win): {avg_top1_win:.2%}")
    print(f"✅ 模型首选马进前三率 (Place): {avg_top1_place:.2%}")
    print(f"✅ 模型预测的前三名中:")
    for count, ratio in hit_counts.items():
        print(f"   - 命中 {count} 匹的场次占比: {ratio:.2%}")
    
    # ==========================================
    # 5. 可视化
    # ==========================================
    plot_analysis(res_df, hit_counts)

def plot_analysis(res_df, hit_counts):
    plt.figure(figsize=(12, 6))
    
    # 饼图：Top 3 命中分布
    plt.subplot(1, 2, 1)
    colors = sns.color_palette("coolwarm", len(hit_counts))
    plt.pie(hit_counts, labels=[f"Hit {i}" for i in hit_counts.index], 
            autopct='%1.1f%%', startangle=90, colors=colors, explode=[0.05]*len(hit_counts))
    plt.title("Model Top-3 Predictions Accuracy Distribution")

    # 柱状图：主要命中率对比
    plt.subplot(1, 2, 2)
    metrics = ['Top-1 Win', 'Top-1 Place', 'Top-3 Avg Precision']
    # Top-3 Avg Precision 是 (命中总数) / (3 * 总场次)
    top3_avg_prec = res_df['top3_hits_count'].sum() / (3 * len(res_df))
    values = [res_df['top1_win_hit'].mean(), res_df['top1_place_hit'].mean(), top3_avg_prec]
    
    sns.barplot(x=metrics, y=values, palette="viridis")
    plt.ylim(0, 1.0)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
    plt.title("Comparison of Hit Rates")

    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "hit_rate_dashboard.png"), dpi=300)
    plt.close()
    print(f"\n✨ 分析图表已保存至: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_accuracy()