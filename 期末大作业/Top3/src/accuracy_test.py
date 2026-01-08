import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置参数 (指向所有模型的结果)
# ==========================================
class Config:
    # 所有模型的预测结果路径
    MODEL_PREDS = {
        "CatBoost": "../result/cat_top3_tuning_result/cat_tuned_preds_top3.csv",
        "LightGBM": "../result/lgbm_top3_tuning_result/lgbm_tuned_preds_top3.csv",
        "XGBoost": "../result/xgb_top3_tuning_result/xgb_tuned_preds_top3.csv",
        "TabNet": "../result/tabnet_top3_result/tabnet_preds_top3.csv",
        "Blending": "../result/blending_top3_tuning_result/final_blended_top3_preds.csv",
        "Stacking": "../result/stacking_top3_result/stacking_final_preds_top3.csv"
    }
    
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
    
    # 读取测试数据和标签
    df_test = pd.read_csv(Config.TEST_DATA_PATH)
    df_label = pd.read_csv(Config.LABEL_PATH)
    
    # 存储所有模型的结果
    all_models_results = []
    
    # 遍历所有模型
    for model_name, pred_path in Config.MODEL_PREDS.items():
        print(f"\n🔍 正在分析模型: {model_name}")
        
        # 检查文件是否存在
        if not os.path.exists(pred_path):
            print(f"   ❌ 文件不存在: {pred_path}")
            continue
        
        # 读取当前模型的概率
        df_prob = pd.read_csv(pred_path)
        
        # 合并成一个分析大表
        analysis_df = pd.DataFrame({
            'race_id': df_test['race_id'],
            'actual_rank': df_test['actual_rank'],
            'pred_prob': df_prob['prob'],
            'is_top3': df_label.iloc[:, -1] # 真实的前三标签
        })
        
        # ==========================================
        # 按场次计算 Top-N 表现
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
        # 统计汇总
        # ==========================================
        total_races = len(res_df)
        avg_top1_place = res_df['top1_place_hit'].mean()
        
        # 计算 Top-3 的分布情况
        hit_counts = res_df['top3_hits_count'].value_counts(normalize=True).sort_index()
        
        # 计算模型认为是前三的马中有多少确实是前三
        # 即：每场比赛预测的前三匹马中，真实前三的数量，然后取平均值
        top3_accuracy = res_df['top3_hits_count'].mean() / 3
        
        print(f"   --- 模型 {model_name} 准确率报告 (总比赛场次: {total_races}) ---")
        print(f"   ✅ 模型首选马进前三率 (Place): {avg_top1_place:.2%}")
        print(f"   ✅ 模型预测的前三名中，真实前三的准确率: {top3_accuracy:.2%}")
        print(f"   ✅ 模型预测的前三名中:")
        for count, ratio in hit_counts.items():
            print(f"      - 命中 {count} 匹的场次占比: {ratio:.2%}")
        
        # 保存当前模型的结果
        all_models_results.append({
            'model_name': model_name,
            'total_races': total_races,
            'top1_place_accuracy': avg_top1_place,
            'top3_accuracy': top3_accuracy,
            'hit_counts': hit_counts
        })
    
    # ==========================================
    # 模型对比可视化
    # ==========================================
    if all_models_results:
        plot_model_comparison(all_models_results)

def plot_model_comparison(all_models_results):
    # 准备对比数据
    model_names = [result['model_name'] for result in all_models_results]
    top1_place_acc = [result['top1_place_accuracy'] for result in all_models_results]
    top3_acc = [result['top3_accuracy'] for result in all_models_results]
    
    # 创建对比图表
    plt.figure(figsize=(16, 8))
    
    # 1. 两种准确率对比柱状图
    plt.subplot(1, 2, 1)
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, top1_place_acc, width, label='模型首选马进前三率', color='#1f77b4')
    bars2 = plt.bar(x + width/2, top3_acc, width, label='模型预测的前三名中，真实前三的准确率', color='#2ca02c')
    
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.title('模型准确率对比')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    # # 2. Top-3 Accuracy 单独对比
    # plt.subplot(1, 2, 2)
    # sorted_indices = np.argsort(top3_acc)[::-1]  # 降序排序
    # sorted_model_names = [model_names[i] for i in sorted_indices]
    # sorted_top3_acc = [top3_acc[i] for i in sorted_indices]
    
    # colors = sns.color_palette("viridis", len(sorted_model_names))
    # bars = plt.bar(sorted_model_names, sorted_top3_acc, color=colors)
    
    # plt.xlabel('模型')
    # plt.ylabel('准确率')
    # plt.title('Top-3 准确率对比 (排序)')
    # plt.xticks(rotation=45, ha='right')
    # plt.ylim(0, 1.0)
    
    # # 添加数值标签
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
    #             f'{height:.2%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(Config.OUTPUT_DIR, "models_accuracy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✨ 分析图表已保存至: {output_path}")
    
    # 输出汇总表格
    print("\n📋 所有模型准确率汇总:")
    print("-" * 80)
    print(f"{'模型名称':<15} {'Top-1 Place准确率':<20} {'Top-3准确率':<15}")
    print("-" * 80)
    for result in all_models_results:
        print(f"{result['model_name']:<15} {result['top1_place_accuracy']:.2%}{'':<10} {result['top3_accuracy']:.2%}")
    print("-" * 80)

if __name__ == "__main__":
    analyze_accuracy()