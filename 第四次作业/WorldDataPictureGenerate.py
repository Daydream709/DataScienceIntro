import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 创建图表保存目录
import os

if not os.path.exists("WorldDataEvaluatePicture"):
    os.makedirs("WorldDataEvaluatePicture")

# 1. 读取数据
paper_counts_df = pd.read_csv("QueryData/RegionRankings/各区域各学科的总论文数.csv")
institution_counts_df = pd.read_csv("QueryData/RegionRankings/各区域在不同学科中的上榜机构数量.csv")
average_ranking_df = pd.read_csv("QueryData/RegionRankings/各区域在不同学科的平均排名.csv")
citation_impact_df = pd.read_csv("QueryData/RegionRankings/各区域在各学科的论文引用影响力.csv")
dominant_subjects_df = pd.read_csv("QueryData/RegionRankings/各区域的优势学科.csv")


# 2. 论文总量分析图表
def plot_paper_counts():
    # 选取主要学科进行展示
    subjects = paper_counts_df["学科"].unique()

    # 创建一个图表，显示各学科论文数量Top 3区域
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for i, subject in enumerate(subjects[:9]):  # 只展示前9个学科
        data = (
            paper_counts_df[paper_counts_df["学科"] == subject]
            .sort_values("总论文数", ascending=False)
            .head(5)
        )
        bars = axes[i].bar(
            data["区域"],
            data["总论文数"] / 10000,
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        )
        axes[i].set_title(f"{subject}", fontsize=12)
        axes[i].set_ylabel("论文数量 (万篇)")
        axes[i].tick_params(axis="x", rotation=45)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            axes[i].annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("WorldDataEvaluatePicture/各学科论文数量Top区域.png", dpi=300, bbox_inches="tight")
    plt.show()


# 3. 上榜机构数量分析图表
def plot_institution_counts():
    # 计算各区域在所有学科中的机构总数
    total_institutions = (
        institution_counts_df.groupby("区域")["上榜机构数"].sum().sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        total_institutions.index,
        total_institutions.values,
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"],
    )

    plt.title("各区域上榜机构总数", fontsize=16)
    plt.ylabel("机构数量", fontsize=14)
    plt.xlabel("区域", fontsize=14)
    plt.xticks(rotation=45)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig("WorldDataEvaluatePicture/各区域上榜机构总数.png", dpi=300, bbox_inches="tight")
    plt.show()


# 4. 平均排名分析图表
def plot_average_ranking():
    # 选择几个重点学科展示平均排名
    key_subjects = [
        "COMPUTER SCIENCE",
        "ECONOMICS & BUSINESS",
        "CLINICAL MEDICINE",
        "ENGINEERING",
        "PHYSICS",
        "MATHEMATICS",
    ]

    fig, ax = plt.subplots(figsize=(14, 8))

    # 为了更好地展示，我们只选择特定学科的数据
    ranking_data = average_ranking_df[average_ranking_df["学科"].isin(key_subjects)]

    # 创建透视表
    pivot_data = ranking_data.pivot(index="学科", columns="区域", values="截尾平均排名")

    # 使用热力图展示
    sns.heatmap(pivot_data, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "平均排名"})
    plt.title("重点学科各区域平均排名热力图（数值越小表现越好）", fontsize=16)
    plt.ylabel("学科", fontsize=14)
    plt.xlabel("区域", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("WorldDataEvaluatePicture/重点学科平均排名热力图.png", dpi=300, bbox_inches="tight")
    plt.show()


# 5. 论文引用影响力分析图表
def plot_citation_impact():
    # 选择几个重点学科展示引用影响力
    key_subjects = [
        "MULTIDISCIPLINARY",
        "CLINICAL MEDICINE",
        "PHYSICS",
        "COMPUTER SCIENCE",
        "MATHEMATICS",
        "SOCIAL SCIENCES, GENERAL",
    ]

    fig, ax = plt.subplots(figsize=(14, 8))

    # 筛选数据
    citation_data = citation_impact_df[citation_impact_df["学科"].isin(key_subjects)]

    # 创建透视表
    pivot_data = citation_data.pivot(index="学科", columns="区域", values="截尾平均citations_per_paper")

    # 使用热力图展示
    sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="Blues", ax=ax, cbar_kws={"label": "平均引用影响力"})
    plt.title("重点学科各区域论文引用影响力热力图（数值越大影响力越强）", fontsize=16)
    plt.ylabel("学科", fontsize=14)
    plt.xlabel("区域", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("WorldDataEvaluatePicture/重点学科引用影响力热力图.png", dpi=300, bbox_inches="tight")
    plt.show()


# 6. 区域优势学科图表
def plot_dominant_subjects():
    plt.figure(figsize=(12, 8))

    # 按区域分组绘制
    regions = dominant_subjects_df["区域"].unique()
    subjects = dominant_subjects_df["学科"].unique()

    # 创建一个包含所有区域和学科的透视表
    pivot_data = dominant_subjects_df.pivot(index="区域", columns="学科", values="平均排名")

    # 使用热力图展示
    sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="Greens", cbar_kws={"label": "平均排名"})
    plt.title("各区域优势学科平均排名", fontsize=16)
    plt.ylabel("区域", fontsize=14)
    plt.xlabel("学科", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("WorldDataEvaluatePicture/各区域优势学科.png", dpi=300, bbox_inches="tight")
    plt.show()


# 运行所有图表函数
if __name__ == "__main__":
    plot_paper_counts()
    plot_institution_counts()
    plot_average_ranking()
    plot_citation_impact()
    plot_dominant_subjects()
    print("所有图表已生成并保存到 'WorldDataEvaluatePicture' 文件夹中")
