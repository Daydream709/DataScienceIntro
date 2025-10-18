import os
import csv
import pymysql
import pandas as pd

# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "tyj810915mysql",
    "database": "esi_data",
    "charset": "utf8mb4",
}


class RegionalAnalysis:
    def __init__(self):
        self.connection = None

    def connect_database(self):
        """连接数据库"""
        try:
            self.connection = pymysql.connect(**DB_CONFIG)
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise

    def close_connection(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("数据库连接已关闭")

    def query_to_csv(self, query, filename):
        """执行查询并将结果保存为CSV文件"""
        try:
            df = pd.read_sql(query, self.connection)
            output_dir = os.path.join(os.path.dirname(__file__), "QueryData", "RegionRankings")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"创建目录: {output_dir}")

            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"查询结果已保存至: {filepath}")
            return df
        except Exception as e:
            print(f"执行查询或保存文件时出错 ({filename}): {e}")
            raise

    def analyze_regional_performance_by_subject_avg_rank(self):
        """各学科领域的区域表现（使用截尾均值，只使用了前3000名）"""
        # 使用子查询方式实现去除头尾10%的数据
        subquery = """
        SELECT 
            s.name AS subject_name,
            c.region AS country_region,
            r.rank_order,
            ROW_NUMBER() OVER (PARTITION BY s.name, c.region ORDER BY r.rank_order) as rn,
            COUNT(*) OVER (PARTITION BY s.name, c.region) as total_count
        FROM rankings r
        JOIN universities u ON r.university_id = u.id
        JOIN countries c ON u.country_id = c.id
        JOIN subjects s ON r.subject_id = s.id
        WHERE r.rank_order <= 3000
        """

        query = f"""
        SELECT 
            t.subject_name AS 学科,
            t.country_region AS 区域,
            COUNT(*) AS 入榜机构数,
            ROUND(AVG(CASE WHEN t.rn > CEIL(t.total_count * 0.1) AND t.rn <= t.total_count - CEIL(t.total_count * 0.1) THEN t.rank_order END), 2) AS 截尾平均排名,
            MIN(t.rank_order) AS 最高排名,
            MAX(t.rank_order) AS 最低排名
        FROM (
            {subquery}
        ) t
        GROUP BY t.subject_name, t.country_region
        HAVING COUNT(*) >= 3 
           AND COUNT(CASE WHEN t.rn > CEIL(t.total_count * 0.1) AND t.rn <= t.total_count - CEIL(t.total_count * 0.1) THEN 1 END) > 0
        ORDER BY t.subject_name, 截尾平均排名
        """
        return self.query_to_csv(query, "各区域在不同学科的平均排名.csv")

    def analyze_regional_institution_count_by_subject(self):
        """各区域在不同学科中的上榜机构数量"""
        query = """
        SELECT 
            c.region AS 区域,
            s.name AS 学科,
            COUNT(*) AS 上榜机构数
        FROM rankings r
        JOIN universities u ON r.university_id = u.id
        JOIN countries c ON u.country_id = c.id
        JOIN subjects s ON r.subject_id = s.id
        GROUP BY c.region, s.name
        ORDER BY c.region, 上榜机构数 DESC
        """
        return self.query_to_csv(query, "各区域在不同学科中的上榜机构数量.csv")

    def analyze_regional_total_metrics(self):
        """各区域各个学科的总论文数"""
        query = """
        SELECT 
            s.name AS 学科,
            c.region AS 区域,
            SUM(r.papers) AS 总论文数
        FROM rankings r
        JOIN universities u ON r.university_id = u.id
        JOIN countries c ON u.country_id = c.id
        JOIN subjects s ON r.subject_id = s.id
        GROUP BY s.name, c.region
        ORDER BY s.name, 总论文数 DESC
        """
        return self.query_to_csv(query, "各区域各学科的总论文数.csv")

    def analyze_top_subjects_by_region(self):
        """各区域的优势学科（平均排名最靠前的学科）"""
        # 这个查询稍微复杂一些，我们需要分步处理
        subquery = """
        SELECT 
            c.region,
            s.name AS subject,
            AVG(r.rank_order) AS avg_rank,
            ROW_NUMBER() OVER (PARTITION BY c.region ORDER BY AVG(r.rank_order)) AS rn
        FROM rankings r
        JOIN universities u ON r.university_id = u.id
        JOIN countries c ON u.country_id = c.id
        JOIN subjects s ON r.subject_id = s.id
        GROUP BY c.region, s.name
        """

        query = f"""
        SELECT 
            t.region AS 区域,
            t.subject AS 学科,
            t.avg_rank AS 平均排名
        FROM ({subquery}) t
        WHERE t.rn = 1
        ORDER BY t.avg_rank
        """

        return self.query_to_csv(query, "各区域的优势学科.csv")

    def analyze_regional_citation_impact(self):
        """各区域在各学科的论文引用影响力（使用cites_per_paper，去除头尾5%数据）"""
        # 使用子查询方式实现去除头尾10%的数据
        subquery = """
        SELECT 
            s.name AS subject_name,
            c.region AS country_region,
            r.cites_per_paper,
            ROW_NUMBER() OVER (PARTITION BY s.name, c.region ORDER BY r.cites_per_paper) as rn,
            COUNT(*) OVER (PARTITION BY s.name, c.region) as total_count
        FROM rankings r
        JOIN universities u ON r.university_id = u.id
        JOIN countries c ON u.country_id = c.id
        JOIN subjects s ON r.subject_id = s.id
        WHERE r.cites_per_paper IS NOT NULL AND r.cites_per_paper >= 0
        """

        query = f"""
        SELECT 
            t.subject_name AS 学科,
            t.country_region AS 区域,
            COUNT(*) AS 机构数,
            ROUND(AVG(t.cites_per_paper), 3) AS `截尾平均citations_per_paper`,
            MIN(t.cites_per_paper) AS 最低值,
            MAX(t.cites_per_paper) AS 最高值
        FROM (
            {subquery}
        ) t
        WHERE t.rn > CEIL(t.total_count * 0.1) 
          AND t.rn <= t.total_count - CEIL(t.total_count * 0.1)
        GROUP BY t.subject_name, t.country_region
        HAVING COUNT(*) >= 3
        ORDER BY t.subject_name, `截尾平均citations_per_paper` DESC
        """
        return self.query_to_csv(query, "各区域在各学科的论文引用影响力.csv")

    def run_all_analyses(self):
        """运行所有分析"""
        try:
            self.connect_database()

            print("开始分析全球不同区域在各个学科中的表现...")

            # 执行各项分析
            self.analyze_regional_performance_by_subject_avg_rank()
            self.analyze_regional_institution_count_by_subject()
            self.analyze_regional_total_metrics()
            self.analyze_top_subjects_by_region()
            self.analyze_regional_citation_impact()

            print("所有分析已完成，结果已保存到CSV文件中。")

        except Exception as e:
            print(f"分析过程中出现错误: {e}")
        finally:
            self.close_connection()


if __name__ == "__main__":
    analyzer = RegionalAnalysis()
    analyzer.run_all_analyses()
