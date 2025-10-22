import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import os

# 设置中文字体支持
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# 创建pictures文件夹（如果不存在）
if not os.path.exists("pictures"):
    os.makedirs("pictures")

# 1. 载入数据
print("=== 第一步：载入数据 ===")
df = pd.read_csv("QueryData/ecnu_rankings.csv")
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 显示前几行数据
print("\n数据预览:")
print(df.head())

# 2. 数据汇总处理
print("\n=== 第二步：数据汇总处理 ===")
# 直接计算新的指标，不进行异常值和缺失值检测
df["篇均引用"] = df["引用次数"] / df["论文数"]

# 3. 数据总览
print("\n=== 第三步：数据总览 ===")
print(f"数据集包含 {len(df)} 个学科进入ESI全球前1%")

# 基本统计信息
print("\n基本统计信息:")
print(df.describe())

# 添加各个指标的平均值与中位数
print("\n各指标平均值与中位数:")
metrics_summary = pd.DataFrame(
    {
        "平均值": [df["排名"].mean(), df["论文数"].mean(), df["引用次数"].mean(), df["篇均引用"].mean()],
        "中位数": [
            df["排名"].median(),
            df["论文数"].median(),
            df["引用次数"].median(),
            df["篇均引用"].median(),
        ],
    },
    index=["排名", "论文数", "引用次数", "篇均引用"],
)
print(metrics_summary)

# 学科数量按领域分类
print("\n学科领域分布:")
subject_categories = {
    "理工科": [
        "CHEMISTRY",
        "MATHEMATICS",
        "PHYSICS",
        "COMPUTER SCIENCE",
        "ENGINEERING",
        "MATERIALS SCIENCE",
        "ENVIRONMENT ECOLOGY",
        "GEOSCIENCES",
    ],
    "社会科学": ["SOCIAL SCIENCES, GENERAL", "PSYCHIATRY PSYCHOLOGY", "ECONOMICS & BUSINESS"],
    "生命科学/医学": [
        "BIOLOGY & BIOCHEMISTRY",
        "MOLECULAR BIOLOGY & GENETICS",
        "NEUROSCIENCE & BEHAVIOR",
        "PLANT & ANIMAL SCIENCE",
        "AGRICULTURAL SCIENCES",
        "CLINICAL MEDICINE",
        "PHARMACOLOGY & TOXICOLOGY",
        "MICROBIOLOGY",
        "IMMUNOLOGY",
    ],
}

# 创建排名分布饼状图
# 设置中文字体支持
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# 数据
labels = [
    "顶尖学科\n(排名≤100)",
    "中上水平学科\n(排名101-500)",
    "中等水平学科\n(排名501-1000)",
    "其他学科\n(排名>1000)",
]
sizes = [1, 9, 5, 2]
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

# 创建饼状图
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12})
plt.axis("equal")  # 确保饼图是圆形
plt.title("华东师范大学ESI学科排名分布", fontsize=16, pad=20)

# 保存图表
plt.tight_layout()
plt.savefig("pictures/ecnu_ranking_distribution_pie_chart.png", dpi=300, bbox_inches="tight")
plt.close()

print("饼状图已保存到 pictures/ecnu_ranking_distribution_pie_chart.png")
category_counts = {}
for category, subjects in subject_categories.items():
    count = len(df[df["学科名称"].isin(subjects)])
    category_counts[category] = count

print(category_counts)


# 设置中文字体支持
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# 读取数据
df = pd.read_csv("ecnu_data.csv")

# 学科分类定义
subject_categories = {
    "理工科": [
        "CHEMISTRY",
        "MATHEMATICS",
        "PHYSICS",
        "COMPUTER SCIENCE",
        "ENGINEERING",
        "MATERIALS SCIENCE",
        "ENVIRONMENT ECOLOGY",
        "GEOSCIENCES",
    ],
    "社会科学": ["SOCIAL SCIENCES, GENERAL", "PSYCHIATRY PSYCHOLOGY", "ECONOMICS & BUSINESS"],
    "生命科学/医学": [
        "BIOLOGY & BIOCHEMISTRY",
        "MOLECULAR BIOLOGY & GENETICS",
        "NEUROSCIENCE & BEHAVIOR",
        "PLANT & ANIMAL SCIENCE",
        "AGRICULTURAL SCIENCES",
        "CLINICAL MEDICINE",
        "PHARMACOLOGY & TOXICOLOGY",
        "MICROBIOLOGY",
        "IMMUNOLOGY",
    ],
}
# 计算每个领域的论文总数并生成柱状图
# 计算各领域论文总数
category_paper_totals = {}
for category, subjects in subject_categories.items():
    category_data = df[df["学科名称"].isin(subjects)]
    total_papers = category_data["论文数"].sum()
    category_paper_totals[category] = total_papers

# 创建柱状图
plt.figure(figsize=(10, 6))
categories = list(category_paper_totals.keys())
paper_totals = list(category_paper_totals.values())

bars = plt.bar(categories, paper_totals, color=["skyblue", "lightgreen", "salmon"])
plt.title("华东师范大学各领域论文总数对比", fontsize=16)
plt.xlabel("学科领域", fontsize=12)
plt.ylabel("论文总数", fontsize=12)

# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, yval + 100, f"{int(yval):,}", ha="center", va="bottom", fontsize=11
    )

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis="y", alpha=0.3)

# 保存图表
plt.tight_layout()
plt.savefig("pictures/ecnu_papers_by_field.png", dpi=300, bbox_inches="tight")
plt.close()

print("各领域论文总数对比图已保存到 pictures/ecnu_papers_by_field.png")
# 4. 特征分析
print("\n=== 第四步：特征分析 ===")

# 4.1 数字特征分析
print("\n=== 数字特征分析 ===")

# 创建数字特征分析图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("华东师范大学学科数字特征分析")

# 排名分布
axes[0, 0].hist(df["排名"], bins=20, alpha=0.7, color="skyblue")
axes[0, 0].set_title("学科排名分布")
axes[0, 0].set_xlabel("排名")
axes[0, 0].set_ylabel("频次")

# 论文数分布
axes[0, 1].hist(df["论文数"], bins=20, alpha=0.7, color="lightgreen")
axes[0, 1].set_title("论文数分布")
axes[0, 1].set_xlabel("论文数")
axes[0, 1].set_ylabel("频次")

# 引用次数分布
axes[1, 0].hist(df["引用次数"], bins=20, alpha=0.7, color="salmon")
axes[1, 0].set_title("引用次数分布")
axes[1, 0].set_xlabel("引用次数")
axes[1, 0].set_ylabel("频次")

# 篇均引用分布
axes[1, 1].hist(df["篇均引用"], bins=20, alpha=0.7, color="gold")
axes[1, 1].set_title("篇均引用分布")
axes[1, 1].set_xlabel("篇均引用")
axes[1, 1].set_ylabel("频次")

plt.tight_layout()
plt.savefig("pictures/ecnu_numeric_features.png", dpi=300, bbox_inches="tight")
plt.close()

# 相关性分析
print("\n相关性矩阵:")
correlation_matrix = df[["排名", "论文数", "引用次数", "篇均引用"]].corr()
print(correlation_matrix)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, square=True)
plt.title("学科指标相关性热力图")
plt.savefig("pictures/ecnu_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# 4.2 类别特征分析
print("\n=== 类别特征分析 ===")

# 学科名称分析
print(f"\n学科名称分布:")
subject_counts = df["学科名称"].value_counts()
print(subject_counts)


# 分析不同领域的表现
print("\n各领域学科表现:")
for category, subjects in subject_categories.items():
    category_data = df[df["学科名称"].isin(subjects)]
    if not category_data.empty:
        print(f"\n{category}:")
        print(f"  学科数量: {len(category_data)}")
        print(f"  平均排名: {category_data['排名'].mean():.1f}")
        print(f"  平均论文数: {category_data['论文数'].mean():,.0f}")
        print(f"  平均引用次数: {category_data['引用次数'].mean():,.0f}")
        print(f"  平均篇均引用: {category_data['篇均引用'].mean():.2f}")

# 各领域表现的可视化
# 领域vs学科数量
plt.figure(figsize=(8, 6))
categories = list(category_counts.keys())
counts = list(category_counts.values())
bars = plt.bar(categories, counts, color=["skyblue", "lightgreen", "salmon"])
plt.title("各领域学科数量分布")
plt.xlabel("领域")
plt.ylabel("学科数量")
# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha="center", va="bottom")
plt.tight_layout()
plt.savefig("pictures/ecnu_category_counts.png", dpi=300, bbox_inches="tight")
plt.close()

# 各领域平均排名对比
category_avg_rankings = {}
for category, subjects in subject_categories.items():
    category_data = df[df["学科名称"].isin(subjects)]
    if not category_data.empty:
        category_avg_rankings[category] = category_data["排名"].mean()

plt.figure(figsize=(8, 6))
categories = list(category_avg_rankings.keys())
avg_rankings = list(category_avg_rankings.values())
bars = plt.bar(categories, avg_rankings, color=["skyblue", "lightgreen", "salmon"])
plt.title("各领域平均排名对比")
plt.xlabel("领域")
plt.ylabel("平均排名")
plt.gca().invert_yaxis()  # 排名越小越好，所以倒置y轴
# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 10, f"{yval:.1f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("pictures/ecnu_category_avg_rankings.png", dpi=300, bbox_inches="tight")
plt.close()

# 各领域平均篇均引用对比
category_avg_cites_per_paper = {}
for category, subjects in subject_categories.items():
    category_data = df[df["学科名称"].isin(subjects)]
    if not category_data.empty:
        category_avg_cites_per_paper[category] = category_data["篇均引用"].mean()

plt.figure(figsize=(8, 6))
categories = list(category_avg_cites_per_paper.keys())
avg_cites_per_paper = list(category_avg_cites_per_paper.values())
bars = plt.bar(categories, avg_cites_per_paper, color=["skyblue", "lightgreen", "salmon"])
plt.title("各领域平均篇均引用对比")
plt.xlabel("领域")
plt.ylabel("平均篇均引用")
# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("pictures/ecnu_category_avg_cites_per_paper.png", dpi=300, bbox_inches="tight")
plt.close()

# 散点图：论文数 vs 排名（按领域着色）
plt.figure(figsize=(10, 8))
colors = ["skyblue", "lightgreen", "salmon"]
for i, (category, subjects) in enumerate(subject_categories.items()):
    category_data = df[df["学科名称"].isin(subjects)]
    if not category_data.empty:
        plt.scatter(
            category_data["论文数"], category_data["排名"], label=category, color=colors[i], alpha=0.7
        )

plt.xlabel("论文数")
plt.ylabel("排名")
plt.title("论文数与排名关系图（按领域分类）")
plt.legend()
plt.gca().invert_yaxis()  # 排名越小越好
plt.grid(True, alpha=0.3)
plt.savefig("pictures/ecnu_papers_vs_rankings.png", dpi=300, bbox_inches="tight")
plt.close()

# 散点图：篇均引用 vs 排名（按领域着色）
plt.figure(figsize=(10, 8))
for i, (category, subjects) in enumerate(subject_categories.items()):
    category_data = df[df["学科名称"].isin(subjects)]
    if not category_data.empty:
        plt.scatter(
            category_data["篇均引用"], category_data["排名"], label=category, color=colors[i], alpha=0.7
        )

plt.xlabel("篇均引用")
plt.ylabel("排名")
plt.title("篇均引用与排名关系图（按领域分类）")
plt.legend()
plt.gca().invert_yaxis()  # 排名越小越好
plt.grid(True, alpha=0.3)
plt.savefig("pictures/ecnu_cites_per_paper_vs_rankings.png", dpi=300, bbox_inches="tight")
plt.close()

# 5. 保存数据到csv文件
print("\n=== 第五步：保存数据 ===")
# 将处理后的数据保存到csv文件
df.to_csv("ecnu_data.csv", index=False, encoding="utf-8")
# 保存统计摘要到csv文件
metrics_summary.to_csv("ecnu_metrics_summary.csv", encoding="utf-8")

print("\n分析完成！")
print("1. 数据已保存到ecnu_data.csv和ecnu_metrics_summary.csv文件")
print("2. 图表已保存到pictures目录")
