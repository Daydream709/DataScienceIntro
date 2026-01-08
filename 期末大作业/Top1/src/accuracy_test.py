import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 路径配置
# ==========================================
class Config:
    # 所有模型的预测结果路径
    MODEL_PREDS = {
        "CatBoost": "../result/cat_tuning_result/cat_tuned_preds.csv",
        "LightGBM": "../result/lgbm_tuning_result/lgbm_tuned_preds.csv",
        "XGBoost": "../result/xgb_tuning_result/xgb_tuned_preds.csv",
        "TabNet": "../result/tabnet_result/tabnet_preds.csv",
        "Blending": "../result/blending_tuning_result/final_blended_preds.csv",
        "Stacking": "../result/stacking_tuning_result/stacking_final_preds.csv"
    }
    
    TEST_DATA_PATH = "../data/X_test_final.csv"
    OUTPUT_DIR = "../result/hit_rate_analysis"

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

def analyze_pure_accuracy():
    print("📊 正在进行 Top-1 命中率深度分析...")
    
    # 读取测试数据
    df_test = pd.read_csv(Config.TEST_DATA_PATH)
    
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
        df_preds = pd.read_csv(pred_path)
        
        # 整合关键列
        analysis_df = pd.DataFrame({
            "race_id": df_test["race_id"].values,
            "actual_rank": df_test["actual_rank"].values,
            "model_prob": df_preds["prob"].values
        })

        # 3. 核心逻辑：找出模型在每场比赛中预测概率最高的马
        # 按 race_id 分组，并提取 model_prob 最大的那一行
        model_favorites = analysis_df.loc[analysis_df.groupby("race_id")["model_prob"].idxmax()].copy()

        # 4. 计算准确率：模型预测第一的马确实是第一名的概率
        total_races = len(model_favorites)
        correct_predictions = model_favorites[model_favorites['actual_rank'] == 1]
        accuracy = len(correct_predictions) / total_races if total_races > 0 else 0
        
        print(f"   --- 模型 {model_name} 准确率报告 (总比赛场次: {total_races}) ---")
        print(f"   ✅ 模型预测第一的马是第一的准确率: {accuracy:.2%}")
        
        # 保存当前模型的结果
        all_models_results.append({
            'model_name': model_name,
            'total_races': total_races,
            'accuracy': accuracy
        })
    
    # ==========================================
    # 模型对比可视化
    # ==========================================
    if all_models_results:
        plot_model_comparison(all_models_results)

def plot_model_comparison(all_models_results):
    # 准备对比数据
    model_names = [result['model_name'] for result in all_models_results]
    accuracies = [result['accuracy'] for result in all_models_results]
    
    # 创建对比图表
    plt.figure(figsize=(12, 6))
    
    # 准确率对比柱状图
    colors = sns.color_palette("viridis", len(model_names))
    bars = plt.bar(model_names, accuracies, color=colors)
    
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.title('模型预测第一的马是第一的准确率对比')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(Config.OUTPUT_DIR, "models_accuracy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✨ 分析图表已保存至: {output_path}")
    
    # 输出汇总表格
    print("\n📋 所有模型准确率汇总:")
    print("-" * 80)
    print(f"{'模型名称':<15} {'预测第一是第一的准确率':<25}")
    print("-" * 80)
    for result in all_models_results:
        print(f"{result['model_name']:<15} {result['accuracy']:.2%}")
    print("-" * 80)

if __name__ == "__main__":
    analyze_pure_accuracy()