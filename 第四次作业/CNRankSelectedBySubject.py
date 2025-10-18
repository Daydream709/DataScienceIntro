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


def save_to_csv_by_subject(column_names, data, folder_name):
    """按学科将查询结果保存为多个CSV文件"""
    try:
        # 确保目标目录存在
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # 按学科分组数据
        subject_data = {}
        for row in data:
            subject_name = row[0]  # 学科名称在第一列
            if subject_name not in subject_data:
                subject_data[subject_name] = []
            subject_data[subject_name].append(row)

        # 为每个学科创建CSV文件
        for subject_name, rows in subject_data.items():
            # 清理文件名中的特殊字符
            filename = "".join(c for c in subject_name if c.isalnum() or c in (" ", "-", "_")).rstrip()
            filename = filename.replace(" ", "_") + ".csv"

            with open(os.path.join(folder_name, filename), "w", newline="", encoding="utf-8-sig") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)
                writer.writerows(rows)
            print(f"学科 '{subject_name}' 的数据已保存到 {folder_name}/{filename}")

        print(f"共生成了 {len(subject_data)} 个学科的CSV文件")
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

        # 显示统计信息
        print(f"\n查询到中国（大陆地区）大学在 {len(set(row[0] for row in results))} 个学科中的表现数据")
        print(f"总共包含 {len(results)} 条记录")

        # 按学科保存为CSV文件
        output_folder = "QueryData/ChinaRankings"
        save_to_csv_by_subject(column_names, results, output_folder)

    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
    finally:
        if connection:
            connection.close()
            print("数据库连接已关闭")


if __name__ == "__main__":
    main()
