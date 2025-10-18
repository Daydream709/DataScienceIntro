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


def query_china_universities_rankings(connection):
    """查询中国（大陆地区）大学在各学科的表现"""
    query = """
    SELECT 
        s.name AS 学科名称,
        u.name AS 大学名称,
        r.rank_order AS 排名,
        r.papers AS 论文数,
        r.citations AS 引用次数,
        r.cites_per_paper AS 每篇论文引用次数
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


def save_to_csv(column_names, data, filename):
    """将查询结果保存为CSV文件"""
    try:
        # 确保QueryData目录存在
        if not os.path.exists("QueryData"):
            os.makedirs("QueryData")

        with open(f"QueryData/{filename}", "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
            writer.writerows(data)
        print(f"查询结果已保存到 QueryData/{filename}")
    except Exception as e:
        print(f"保存文件时出现错误: {e}")
        raise


def main():
    """主函数"""
    connection = None
    try:
        # 连接数据库
        connection = connect_database()

        # 执行查询
        column_names, results = query_china_universities_rankings(connection)

        # 显示部分查询结果
        print("\n中国（大陆地区）大学在各学科的表现 (前20条记录):")
        print("-" * 100)
        print(
            f"{'学科名称':<25} {'大学名称':<30} {'排名':<8} {'论文数':<10} {'引用次数':<12} {'每篇论文引用次数':<15}"
        )
        print("-" * 100)
        for row in results[:20]:
            print(f"{row[0]:<25} {row[1]:<30} {row[2]:<8} {row[3]:<10} {row[4]:<12} {row[5]:<15}")

        if len(results) > 20:
            print(f"\n... 还有 {len(results) - 20} 条记录未显示，请查看生成的CSV文件获取完整数据")

        # 保存为CSV文件
        output_file = "china_universities_rankings.csv"
        save_to_csv(column_names, results, output_file)

    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
    finally:
        if connection:
            connection.close()
            print("数据库连接已关闭")


if __name__ == "__main__":
    main()
