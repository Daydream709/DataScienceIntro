import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 创建输出目录
output_dir = "CNDataEvaluatePicture"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载数据
df = pd.read_csv(
    "d:\\code\\DataScienceIntro\\第四次作业\\QueryData\\china_subjects_comprehensive_analysis.csv",
    encoding="utf-8",
)
df.set_index("学科名称", inplace=True)

# 确保列名正确
df.columns = [
    "最高排名",
    "进入前10名机构数",
    "进入前50名机构数",
    "进入前100名机构数",
    "上榜机构总数",
    "总论文数",
    "总引用次数",
    "中国平均每篇论文引用次数",
    "全球平均每篇论文引用次数",
]

# 排序：按“进入前100名机构数”降序，再按“最高排名”升序
df = df.sort_values(by=["进入前100名机构数", "最高排名"], ascending=[False, True])

# ==================== 图表1：各学科最高排名（越低越好） ====================
plt.figure(figsize=(14, 6))
plt.bar(df.index, df["最高排名"], color="blue", edgecolor="black")
plt.title("各学科最高排名（越低越好）")
plt.ylabel("最高排名")
plt.xlabel("学科")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "chart_01_highest_ranking.png"), dpi=100, bbox_inches="tight")
plt.close()

# ==================== 图表2：各学科上榜机构总数 ====================
plt.figure(figsize=(14, 6))
plt.bar(df.index, df["上榜机构总数"], color="skyblue", edgecolor="black")
plt.title("各学科上榜机构总数")
plt.ylabel("机构数量")
plt.xlabel("学科")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "chart_02_total_institutions.png"), dpi=100, bbox_inches="tight")
plt.close()

# ==================== 图表3：各学科顶尖机构数量对比（前10/50/100） ====================
top10_subj = df.nlargest(8, "进入前100名机构数").index.tolist()

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(top10_subj))
width = 0.25

ax.bar(x - width, df.loc[top10_subj, "进入前10名机构数"], width, label="前10名", color="green")
ax.bar(x, df.loc[top10_subj, "进入前50名机构数"], width, label="前50名", color="orange")
ax.bar(x + width, df.loc[top10_subj, "进入前100名机构数"], width, label="前100名", color="lightblue")

ax.set_xlabel("学科")
ax.set_ylabel("机构数量")
ax.set_title("各学科顶尖机构数量对比（前10/50/100）")
ax.set_xticks(x)
ax.set_xticklabels(top10_subj, rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "chart_03_top_institutions_comparison.png"), dpi=100, bbox_inches="tight"
)
plt.close()

# ==================== 图表4：论文产出与引用情况（前10学科） ====================
top10_by_papers = df.nlargest(10, "总论文数").index.tolist()

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(
    top10_by_papers,
    df.loc[top10_by_papers, "总论文数"] / 10000,
    marker="o",
    label="总论文数(万篇)",
    color="blue",
)
ax1.set_ylabel("论文数(万篇)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(
    top10_by_papers,
    df.loc[top10_by_papers, "总引用次数"] / 1000000,
    marker="s",
    label="总引用次数(百万次)",
    color="red",
)
ax2.set_ylabel("引用次数(百万次)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("论文产出与引用情况（前10学科）")
plt.xlabel("学科")
plt.xticks(rotation=45)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "chart_04_publication_and_citation.png"), dpi=100, bbox_inches="tight")
plt.close()

# ==================== 图表5：平均每篇论文引用次数对比（中国 vs 全球） ====================
plt.figure(figsize=(14, 6))
x = np.arange(len(df.index))
width = 0.35

plt.bar(x - width / 2, df["中国平均每篇论文引用次数"], width, label="中国平均", color="red")
plt.bar(x + width / 2, df["全球平均每篇论文引用次数"], width, label="全球平均", color="blue")

plt.title("平均每篇论文引用次数对比")
plt.ylabel("平均引用次数")
plt.xlabel("学科")
plt.xticks(x, df.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "chart_05_average_citations_comparison.png"), dpi=100, bbox_inches="tight"
)
plt.close()

# ==================== 图表6：每篇论文平均引用次数（仅中国） ====================
plt.figure(figsize=(14, 6))
plt.bar(df.index, df["中国平均每篇论文引用次数"], color="purple", edgecolor="black")
plt.title("每篇论文平均引用次数（中国）")
plt.ylabel("平均引用次数")
plt.xlabel("学科")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "chart_06_china_average_citations.png"), dpi=100, bbox_inches="tight")
plt.close()

# ==================== 图表7：顶尖机构占比（前10名 / 总上榜数） ====================
df["顶尖机构占比"] = df["进入前10名机构数"] / df["上榜机构总数"]
plt.figure(figsize=(14, 6))
plt.bar(df.index, df["顶尖机构占比"], color="orange", edgecolor="black")
plt.title("顶尖机构占比（前10名/总上榜数）")
plt.ylabel("占比")
plt.xlabel("学科")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "chart_07_top_institution_ratio.png"), dpi=100, bbox_inches="tight")
plt.close()

# ==================== 图表8：综合实力雷达图（前8学科） ====================
top8 = df.nlargest(8, "总论文数").index.tolist()

# 标准化三个指标
metrics = ["最高排名", "进入前10名机构数", "总引用次数"]
normalized_data = df.loc[top8, metrics].copy()

# 最高排名越小越好 → 反向处理
normalized_data["最高排名"] = normalized_data["最高排名"].max() - normalized_data["最高排名"]

# 归一化到 [0,1]
normalized_data = (normalized_data - normalized_data.min()) / (normalized_data.max() - normalized_data.min())

# 雷达图
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for idx, subject in enumerate(top8):
    values = normalized_data.loc[subject].values.tolist()
    values += values[:1]  # 闭合
    ax.plot(angles, values, label=subject)
    ax.fill(angles, values, alpha=0.25)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
plt.title("综合实力雷达图（前8学科）")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.savefig(os.path.join(output_dir, "chart_08_radar_chart.png"), dpi=100, bbox_inches="tight")
plt.close()

# ==================== 图表9：国际竞争力分析（中国/全球平均引用次数） ====================
df["竞争力比率"] = df["中国平均每篇论文引用次数"] / df["全球平均每篇论文引用次数"]

plt.figure(figsize=(14, 6))
bars = plt.bar(
    df.index,
    df["竞争力比率"],
    color=["green" if x >= 1 else "red" for x in df["竞争力比率"]],
    edgecolor="black",
)

plt.axhline(y=1, color="black", linestyle="--", linewidth=1)
plt.title("国际竞争力分析（中国/全球平均引用次数）")
plt.ylabel("竞争力比率")
plt.xlabel("学科")
plt.xticks(rotation=45)

# 添加领先学科数量标注
leading_count = sum(df["竞争力比率"] >= 1)
plt.text(
    0.02,
    0.98,
    f"领先全球平均水平的学科数: {leading_count}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"),
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "chart_09_competitiveness_analysis.png"), dpi=100, bbox_inches="tight")
plt.close()

print(f"所有图表已成功保存至：{output_dir}")
