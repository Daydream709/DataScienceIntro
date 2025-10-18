import pymysql
import csv
import os
import statistics
from collections import defaultdict

# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "tyj810915mysql",
    "database": "esi_data",
    "charset": "utf8mb4",
}


def connect_database():
    """连接数据库"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        print("数据库连接成功")
        return connection
    except Exception as e:
        print(f"数据库连接失败: {e}")
        raise


def query_china_subject_performance(connection):
    """查询中国（大陆地区）在各学科的表现"""
    query = """
    SELECT 
        s.name AS 学科名称,
        r.rank_order AS 排名,
        r.papers AS 论文数,
        r.citations AS 引用次数,
        r.cites_per_paper AS 每篇论文引用次数,
        u.name AS 大学名称
    FROM rankings r
    JOIN universities u ON r.university_id = u.id
    JOIN countries c ON u.country_id = c.id
    JOIN subjects s ON r.subject_id = s.id
    WHERE c.name = 'CHINA MAINLAND'
    ORDER BY s.name, r.rank_order;
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            return column_names, results
    except Exception as e:
        print(f"查询过程中出现错误: {e}")
        raise


def query_global_cites_per_paper_average(connection):
    """查询全球各学科的平均 cites_per_paper"""
    # 使用截尾平均值（去除头尾10%的数据）以获得更准确的全球平均值
    subquery = """
    SELECT 
        s.name AS subject_name,
        r.cites_per_paper,
        ROW_NUMBER() OVER (PARTITION BY s.name ORDER BY r.cites_per_paper) as rn,
        COUNT(*) OVER (PARTITION BY s.name) as total_count
    FROM rankings r
    JOIN universities u ON r.university_id = u.id
    JOIN countries c ON u.country_id = c.id
    JOIN subjects s ON r.subject_id = s.id
    WHERE r.cites_per_paper IS NOT NULL AND r.cites_per_paper >= 0
    """

    query = f"""
    SELECT 
        t.subject_name AS 学科名称,
        ROUND(AVG(t.cites_per_paper), 3) AS 全球平均每篇论文引用次数
    FROM (
        {subquery}
    ) t
    WHERE t.rn > CEIL(t.total_count * 0.1) 
      AND t.rn <= t.total_count - CEIL(t.total_count * 0.1)
    GROUP BY t.subject_name
    ORDER BY t.subject_name
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            return dict(results)  # 返回字典格式 {学科名称: 全球平均每篇论文引用次数}
    except Exception as e:
        print(f"查询全球平均引用次数时出现错误: {e}")
        raise


def analyze_china_subject_performance(data, global_cites_data):
    """分析中国在各学科的表现"""
    # 按学科分组数据
    subject_data = defaultdict(list)

    for row in data:
        subject_name = row[0]
        rank = row[1]
        papers = row[2]
        citations = row[3]
        cites_per_paper = row[4]
        university = row[5]

        subject_data[subject_name].append(
            {
                "rank": rank,
                "papers": papers,
                "citations": citations,
                "cites_per_paper": cites_per_paper,
                "university": university,
            }
        )

    # 计算每个学科的统计数据
    subject_stats = []
    for subject, rankings in subject_data.items():
        top_rank = min(r["rank"] for r in rankings)
        top_10_count = sum(1 for r in rankings if r["rank"] <= 10)
        top_50_count = sum(1 for r in rankings if r["rank"] <= 50)
        top_100_count = sum(1 for r in rankings if r["rank"] <= 100)
        total_institutions = len(rankings)

        total_papers = sum(r["papers"] for r in rankings)
        total_citations = sum(r["citations"] for r in rankings)
        avg_cites_per_paper = statistics.mean([r["cites_per_paper"] for r in rankings]) if rankings else 0

        # 获取全球平均引用次数
        global_avg_cites = global_cites_data.get(subject, 0)

        subject_stats.append(
            {
                "学科名称": subject,
                "最高排名": top_rank,
                "进入前10名机构数": top_10_count,
                "进入前50名机构数": top_50_count,
                "进入前100名机构数": top_100_count,
                "上榜机构总数": total_institutions,
                "总论文数": total_papers,
                "总引用次数": total_citations,
                "中国平均每篇论文引用次数": round(avg_cites_per_paper, 3),
                "全球平均每篇论文引用次数": global_avg_cites,
            }
        )

    # 按进入前100名机构数排序
    subject_stats.sort(key=lambda x: (-x["进入前100名机构数"], x["最高排名"]))
    return subject_stats


def save_to_csv(data, filename, fieldnames):
    """将数据保存为CSV文件"""
    try:
        # 确保QueryData目录存在
        output_dir = "QueryData"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"查询结果已保存到 {filepath}")
    except Exception as e:
        print(f"保存文件时出现错误: {e}")
        raise


def main():
    """主函数"""
    connection = None
    try:
        # 连接数据库
        connection = connect_database()

        # 查询中国在各学科的表现
        china_column_names, china_results = query_china_subject_performance(connection)

        # 查询全球各学科的平均引用次数
        global_cites_data = query_global_cites_per_paper_average(connection)

        # 显示基本统计信息
        print(f"\n查询到中国（大陆地区）在 {len(set(row[0] for row in china_results))} 个学科中的表现数据")
        print(f"总共包含 {len(china_results)} 条记录")

        # 分析中国在各学科的表现
        print("\n正在分析中国在各学科的表现...")
        subject_analysis = analyze_china_subject_performance(china_results, global_cites_data)

        print("\n中国在各学科的表现:")
        print("-" * 130)
        print(
            f"{'学科名称':<25} {'最高排名':<8} {'前10名数':<8} {'前50名数':<8} {'前100名数':<9} {'上榜总数':<8} {'中国平均引用':<12} {'全球平均引用':<12}"
        )
        print("-" * 130)
        for subj in subject_analysis:
            print(
                f"{subj['学科名称']:<25} {subj['最高排名']:<8} {subj['进入前10名机构数']:<8} "
                f"{subj['进入前50名机构数']:<8} {subj['进入前100名机构数']:<9} {subj['上榜机构总数']:<8} "
                f"{subj['中国平均每篇论文引用次数']:<12} {subj['全球平均每篇论文引用次数']:<12}"
            )

        # 保存综合分析结果到CSV文件
        print("\n正在保存分析结果...")

        # 保存中国各学科表现分析（综合文件）
        subject_fieldnames = [
            "学科名称",
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
        save_to_csv(subject_analysis, "china_subjects_comprehensive_analysis.csv", subject_fieldnames)

        print(f"\n分析完成！生成的文件:")
        print(f"QueryData/china_subjects_comprehensive_analysis.csv - 中国各学科综合表现分析")

    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
    finally:
        if connection:
            connection.close()
            print("数据库连接已关闭")


if __name__ == "__main__":
    main()
