import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_all_data():
    """加载所有学科数据"""
    data_dir = "download"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    all_data = {}
    for csv_file in csv_files:
        subject_name = csv_file.replace(".csv", "")
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(file_path, skiprows=1, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, skiprows=1, encoding="ISO-8859-1")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, skiprows=1, encoding="utf-8-sig")

        # 清理列名
        df.columns = ["Rank", "Institution", "Country", "Documents", "Cites", "CitesPerPaper", "TopPapers"]
        all_data[subject_name] = df

    return all_data


def calculate_university_indicators(all_data):
    """计算各大学的指标"""
    # 获取所有大学名称
    all_universities = set()
    for df in all_data.values():
        all_universities.update(df["Institution"].dropna().unique())

    university_indicators = {}

    for university in all_universities:
        # 初始化指标
        indicators = {
            "university_name": university,
            "total_subjects": 0,
            "top_subjects": 0,  # 排名<=100
            "high_subjects": 0,  # 排名101-500
            "medium_subjects": 0,  # 排名501-1000
            "other_subjects": 0,  # 排名>1000
            "total_documents": 0,
            "total_cites": 0,
            "avg_cites_per_paper": 0,
            "subjects_in_top_500": [],
        }

        total_cites_per_paper = 0
        count_cites_per_paper = 0

        # 遍历所有学科数据
        for subject, df in all_data.items():
            uni_data = df[df["Institution"] == university]
            if not uni_data.empty:
                indicators["total_subjects"] += 1

                rank = uni_data["Rank"].iloc[0]
                # 确保rank是数字类型
                if pd.notna(rank):
                    rank = pd.to_numeric(rank, errors="coerce")

                if pd.notna(rank):
                    if rank <= 100:
                        indicators["top_subjects"] += 1
                    elif rank <= 500:
                        indicators["high_subjects"] += 1
                    elif rank <= 1000:
                        indicators["medium_subjects"] += 1
                    else:
                        indicators["other_subjects"] += 1

                    if rank <= 500:
                        indicators["subjects_in_top_500"].append(subject)
                else:
                    # 如果rank是NaN，我们将它归类为"其他"
                    indicators["other_subjects"] += 1
                    continue  # 跳过这个学科的其他处理

                documents = uni_data["Documents"].iloc[0]
                cites = uni_data["Cites"].iloc[0]
                cites_per_paper = uni_data["CitesPerPaper"].iloc[0]

                # 确保数值列也被正确解析
                if pd.notna(documents):
                    documents = pd.to_numeric(str(documents).replace(",", ""), errors="coerce")
                if pd.notna(cites):
                    cites = pd.to_numeric(str(cites).replace(",", ""), errors="coerce")
                if pd.notna(cites_per_paper):
                    cites_per_paper = pd.to_numeric(cites_per_paper, errors="coerce")

                if pd.notna(documents):
                    indicators["total_documents"] += documents
                if pd.notna(cites):
                    indicators["total_cites"] += cites
                if pd.notna(cites_per_paper):
                    total_cites_per_paper += cites_per_paper
                    count_cites_per_paper += 1

        # 计算平均篇均引用次数
        if count_cites_per_paper > 0:
            indicators["avg_cites_per_paper"] = total_cites_per_paper / count_cites_per_paper

        university_indicators[university] = indicators

    return university_indicators


def prepare_clustering_data(university_indicators):
    """准备聚类数据"""
    data_for_clustering = []
    university_names = []

    for uni_name, indicators in university_indicators.items():
        # 跳过数据不完整的大学
        if indicators["total_subjects"] == 0:
            continue

        university_names.append(uni_name)
        data_for_clustering.append(
            [
                indicators["total_subjects"],
                indicators["top_subjects"],
                indicators["high_subjects"],
                indicators["medium_subjects"],
                indicators["other_subjects"],
                indicators["total_documents"],
                indicators["total_cites"],
                indicators["avg_cites_per_paper"],
                len(indicators["subjects_in_top_500"]),
            ]
        )

    # 转换为DataFrame
    columns = [
        "total_subjects",
        "top_subjects",
        "high_subjects",
        "medium_subjects",
        "other_subjects",
        "total_documents",
        "total_cites",
        "avg_cites_per_paper",
        "subjects_in_top_500_count",
    ]

    df_clustering = pd.DataFrame(data_for_clustering, columns=columns)
    df_clustering["university_name"] = university_names

    return df_clustering


def perform_clustering(df_clustering, n_clusters=5):
    """执行聚类分析"""
    # 提取数值特征用于聚类
    feature_columns = [
        "total_subjects",
        "top_subjects",
        "high_subjects",
        "medium_subjects",
        "other_subjects",
        "total_documents",
        "total_cites",
        "avg_cites_per_paper",
        "subjects_in_top_500_count",
    ]

    X = df_clustering[feature_columns]

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 执行K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 将聚类结果添加到DataFrame
    df_clustering["cluster"] = cluster_labels

    return df_clustering, scaler, kmeans


def analyze_clusters(df_clustering):
    """分析聚类结果"""
    print("聚类结果分析:")
    print("=" * 50)

    for cluster_id in sorted(df_clustering["cluster"].unique()):
        cluster_data = df_clustering[df_clustering["cluster"] == cluster_id]
        print(f"\n簇 {cluster_id} (包含 {len(cluster_data)} 所大学):")
        print("-" * 30)

        # 显示该簇的统计信息
        print(f"平均学科数量: {cluster_data['total_subjects'].mean():.2f}")
        print(f"平均顶尖学科数(≤100): {cluster_data['top_subjects'].mean():.2f}")
        print(f"平均中上学科数(101-500): {cluster_data['high_subjects'].mean():.2f}")
        print(f"平均中等学科数(501-1000): {cluster_data['medium_subjects'].mean():.2f}")
        print(f"平均论文总数: {cluster_data['total_documents'].mean():.0f}")
        print(f"平均引用次数: {cluster_data['total_cites'].mean():.0f}")
        print(f"平均篇均引用次数: {cluster_data['avg_cites_per_paper'].mean():.2f}")
        print(f"平均进入前500名的学科数: {cluster_data['subjects_in_top_500_count'].mean():.2f}")

        # 显示该簇中的部分大学
        print(f"该簇中的大学示例:")
        for uni in cluster_data["university_name"].head(5):
            print(f"  - {uni}")


def find_similar_to_ecnu(df_clustering):
    """找出与华东师范大学相似的大学"""
    # 查找华东师范大学所在的簇
    ecnu_data = df_clustering[
        (df_clustering["university_name"] == "EAST CHINA NORMAL UNIVERSITY")
        | (df_clustering["university_name"] == "华东师范大学")
    ]

    if ecnu_data.empty:
        print("未找到华东师范大学的数据")
        return

    ecnu_cluster = ecnu_data["cluster"].iloc[0]
    print(f"\n华东师范大学被分配到簇 {ecnu_cluster}")

    # 获取同一簇中的所有大学
    similar_universities = df_clustering[df_clustering["cluster"] == ecnu_cluster]

    print(f"\n与华东师范大学相似的大学 (同一簇中的大学):")
    print("=" * 50)
    for i, (idx, row) in enumerate(similar_universities.iterrows(), 1):
        print(f"{i}. {row['university_name']}")

    # 保存相似大学到单独的CSV文件
    if not os.path.exists("second_model_results"):
        os.makedirs("second_model_results")
    similar_universities.to_csv("second_model_results/ecnu_similar_universities.csv", index=False, encoding="utf-8-sig")
    print(f"\n与华东师范大学相似的大学已保存到 second_model_results/ecnu_similar_universities.csv")

    return similar_universities



def visualize_clusters(df_clustering):
    """可视化聚类结果"""
    # 确保保存目录存在
    if not os.path.exists("second_model_results"):
        os.makedirs("second_model_results")

    # 使用PCA降维到2D进行可视化
    feature_columns = [
        "total_subjects",
        "top_subjects",
        "high_subjects",
        "medium_subjects",
        "other_subjects",
        "total_documents",
        "total_cites",
        "avg_cites_per_paper",
        "subjects_in_top_500_count",
    ]

    X = df_clustering[feature_columns]

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 创建可视化图表
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_clustering["cluster"], cmap="viridis", alpha=0.7)

    # 标注华东师范大学
    ecnu_indices = df_clustering[
        (df_clustering["university_name"] == "EAST CHINA NORMAL UNIVERSITY")
        | (df_clustering["university_name"] == "华东师范大学")
    ].index

    if len(ecnu_indices) > 0:
        ecnu_idx = ecnu_indices[0]
        plt.scatter(X_pca[ecnu_idx, 0], X_pca[ecnu_idx, 1], c="red", s=100, marker="*", label="华东师范大学")

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.title("大学聚类结果可视化 (PCA降维)")
    plt.colorbar(scatter, label="Cluster")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("second_model_results/clustering_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 绘制各簇特征雷达图
    cluster_features = df_clustering.groupby("cluster")[feature_columns].mean()

    # 标准化特征用于可视化
    scaler = StandardScaler()
    cluster_features_scaled = scaler.fit_transform(cluster_features)
    cluster_features_scaled = pd.DataFrame(
        cluster_features_scaled, columns=feature_columns, index=cluster_features.index
    )

    # 绘制雷达图
    plt.figure(figsize=(12, 10))
    categories = feature_columns

    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]  # 闭合图形

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection="polar"))

    for cluster_id in cluster_features_scaled.index:
        values = cluster_features_scaled.loc[cluster_id].tolist()
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, "o-", linewidth=2, label=f"簇 {cluster_id}")
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(-3, 3)
    plt.title("各簇特征雷达图", size=16, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig("second_model_results/cluster_radar_chart.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """主函数"""
    print("开始对 ESI 数据进行聚类分析...")

    # 确保保存目录存在
    if not os.path.exists("second_model_results"):
        os.makedirs("second_model_results")

    # 加载所有数据
    print("正在加载数据...")
    all_data = load_all_data()
    print(f"共加载了 {len(all_data)} 个学科的数据")

    # 计算各大学指标
    print("正在计算各大学指标...")
    university_indicators = calculate_university_indicators(all_data)
    print(f"共处理了 {len(university_indicators)} 所大学的数据")

    # 准备聚类数据
    print("正在准备聚类数据...")
    df_clustering = prepare_clustering_data(university_indicators)

    # 执行聚类
    print("正在进行聚类分析...")
    df_clustering, scaler, kmeans = perform_clustering(df_clustering, n_clusters=5)

    # 分析聚类结果
    analyze_clusters(df_clustering)

    # 查找与华东师范大学相似的大学
    similar_unis = find_similar_to_ecnu(df_clustering)

    # 可视化聚类结果
    print("正在生成可视化图表...")
    visualize_clusters(df_clustering)

    # 保存结果
    df_clustering.to_csv("second_model_results/clustering_results.csv", index=False, encoding="utf-8-sig")
    print("\n聚类分析完成，结果已保存到 second_model_results/clustering_results.csv")
    print(
        "可视化图表已保存到 second_model_results/clustering_visualization.png 和 second_model_results/cluster_radar_chart.png"
    )


if __name__ == "__main__":
    main()
