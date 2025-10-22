import pymysql
import csv
import os

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


def analyze_ecnu_indicators(connection):
    """分析华东师范大学的7个指标并返回结果"""
    indicators = {}

    try:
        with connection.cursor() as cursor:
            # 1. 进入 ESI 前 1%的学科数量
            query1 = """
            SELECT COUNT(*) AS 学科总数
            FROM rankings r
            JOIN universities u ON r.university_id = u.id
            WHERE u.name = 'EAST CHINA NORMAL UNIVERSITY' OR u.name = '华东师范大学';
            """
            cursor.execute(query1)
            indicators["进入ESI前1%的学科数量"] = cursor.fetchone()[0]

            # 2. 各学科排名分布情况（顶尖/中上/中等/其他）
            query2 = """
            SELECT 
                SUM(CASE WHEN r.rank_order <= 100 THEN 1 ELSE 0 END) AS 顶尖学科数,
                SUM(CASE WHEN r.rank_order > 100 AND r.rank_order <= 500 THEN 1 ELSE 0 END) AS 中上学科数,
                SUM(CASE WHEN r.rank_order > 500 AND r.rank_order <= 1000 THEN 1 ELSE 0 END) AS 中等学科数,
                SUM(CASE WHEN r.rank_order > 1000 THEN 1 ELSE 0 END) AS 其他学科数
            FROM rankings r
            JOIN universities u ON r.university_id = u.id
            WHERE u.name = 'EAST CHINA NORMAL UNIVERSITY' OR u.name = '华东师范大学';
            """
            cursor.execute(query2)
            distribution = cursor.fetchone()
            indicators["顶尖学科数(≤100)"] = distribution[0]
            indicators["中上学科数(101-500)"] = distribution[1]
            indicators["中等学科数(501-1000)"] = distribution[2]
            indicators["其他学科数(>1000)"] = distribution[3]

            # 新增指标：获取进入前500名的学科名称
            query_top500_subjects = """
            SELECT s.name
            FROM rankings r
            JOIN universities u ON r.university_id = u.id
            JOIN subjects s ON r.subject_id = s.id
            WHERE (u.name = 'EAST CHINA NORMAL UNIVERSITY' OR u.name = '华东师范大学')
            AND r.rank_order <= 500
            ORDER BY r.rank_order;
            """
            cursor.execute(query_top500_subjects)
            top500_subjects = cursor.fetchall()
            indicators["进入前500名的学科"] = [subject[0] for subject in top500_subjects]

            # 3. 学科领域覆盖范围（理工科、文科、医科等比例）
            query3 = """
            SELECT 
                SUM(CASE 
                    WHEN s.name IN ('ENGINEERING', 'MATERIALS SCIENCE', 'CHEMISTRY', 'COMPUTER SCIENCE', 'MATHEMATICS', 'PHYSICS') 
                    THEN 1 ELSE 0 
                END) AS 理工科,
                SUM(CASE 
                    WHEN s.name IN ('ECONOMICS & BUSINESS', 'SOCIAL SCIENCES, GENERAL', 'PSYCHIATRY_PSYCHOLOGY') 
                    THEN 1 ELSE 0 
                END) AS 文科,
                SUM(CASE 
                    WHEN s.name IN ('CLINICAL MEDICINE', 'IMMUNOLOGY', 'MICROBIOLOGY', 'PHARMACOLOGY & TOXICOLOGY') 
                    THEN 1 ELSE 0 
                END) AS 医科
            FROM rankings r
            JOIN universities u ON r.university_id = u.id
            JOIN subjects s ON r.subject_id = s.id
            WHERE u.name = 'EAST CHINA NORMAL UNIVERSITY' OR u.name = '华东师范大学';
            """
            cursor.execute(query3)
            field_coverage = cursor.fetchone()
            indicators["理工科领域学科数"] = field_coverage[0]
            indicators["文科领域学科数"] = field_coverage[1]
            indicators["医科领域学科数"] = field_coverage[2]

            # 4. 研究实力指标
            query4 = """
            SELECT 
                SUM(r.papers) AS 论文总数,
                SUM(r.citations) AS 引用次数总数,
                ROUND(SUM(r.citations) / SUM(r.papers), 2) AS 平均每篇论文引用次数
            FROM rankings r
            JOIN universities u ON r.university_id = u.id
            WHERE u.name = 'EAST CHINA NORMAL UNIVERSITY' OR u.name = '华东师范大学';
            """
            cursor.execute(query4)
            research_metrics = cursor.fetchone()
            indicators["论文总数"] = research_metrics[0]
            indicators["引用次数总数"] = research_metrics[1]
            indicators["平均每篇论文引用次数"] = research_metrics[2]

    except Exception as e:
        print(f"分析指标过程中出现错误: {e}")
        raise

    return indicators


def save_indicators_to_csv(indicators, filename):
    """将指标保存为CSV文件"""
    try:
        # 确保QueryData目录存在
        output_dir = "QueryData"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["指标名称", "指标值"])
            for key, value in indicators.items():
                # 特殊处理进入前500名的学科列表
                if key == "进入前500名的学科":
                    subjects_str = "; ".join(value)
                    writer.writerow([key, subjects_str])
                else:
                    writer.writerow([key, value])
        print(f"指标已保存到 {filepath}")
    except Exception as e:
        print(f"保存文件时出现错误: {e}")
        raise


def main():
    """主函数"""
    connection = None
    try:
        # 连接数据库
        connection = connect_database()

        # 分析华东师范大学的7个指标
        indicators = analyze_ecnu_indicators(connection)

        # 显示指标结果
        print("华东师范大学关键指标:")
        print("-" * 50)
        # 按您的指标分类显示
        print("1. 进入ESI前1%的学科数量:")
        print(f"   {indicators['进入ESI前1%的学科数量']}")

        print("\n2. 各学科排名分布情况:")
        print(f"   顶尖(≤100): {indicators['顶尖学科数(≤100)']}")
        print(f"   中上(101-500): {indicators['中上学科数(101-500)']}")
        print(f"   中等(501-1000): {indicators['中等学科数(501-1000)']}")
        print(f"   其他(>1000): {indicators['其他学科数(>1000)']}")

        # 显示进入前500名的学科
        print("\n2.5. 进入前500名的学科:")
        top500_subjects = indicators["进入前500名的学科"]
        if top500_subjects:
            for subject in top500_subjects:
                print(f"   {subject}")
        else:
            print("   无学科进入前500名")

        print("\n3. 学科领域覆盖范围:")
        print(f"   理工科: {indicators['理工科领域学科数']}")
        print(f"   文科: {indicators['文科领域学科数']}")
        print(f"   医科: {indicators['医科领域学科数']}")

        print("\n4. 研究实力:")
        print(f"   论文总数: {indicators['论文总数']:,}")
        print(f"   引用次数总数: {indicators['引用次数总数']:,}")
        print(f"   平均每篇论文引用次数: {indicators['平均每篇论文引用次数']}")

        # 保存指标到CSV文件
        save_indicators_to_csv(indicators, "ecnu_indicators.csv")

        print("\n指标保存完成！")

    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
    finally:
        if connection:
            connection.close()
            print("数据库连接已关闭")


if __name__ == "__main__":
    main()
