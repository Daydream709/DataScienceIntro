import os
import csv
import pymysql
from pathlib import Path


def create_horse_table(cursor):
    """创建 horse 表"""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS horse (
            horse_name VARCHAR(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci PRIMARY KEY,
            gender VARCHAR(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    )


def insert_horses_from_csv(cursor, csv_file_path):
    """从CSV文件中读取马匹信息并插入到horse表中"""
    # 直接使用UTF-8编码读取文件
    try:
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                horse_name = row["馬名"]
                sex_age = row["性齢"]  # 性齢列包含性别和年龄

                # 提取性别（第一个字符）
                gender = sex_age[0] if sex_age else None

                # 检查是否已存在该马匹
                cursor.execute("SELECT COUNT(*) FROM horse WHERE horse_name = %s", (horse_name,))
                count = cursor.fetchone()[0]

                if count == 0:  # 如果不存在，则插入
                    cursor.execute(
                        "INSERT INTO horse (horse_name, gender) VALUES (%s, %s)", (horse_name, gender)
                    )

    except UnicodeDecodeError:
        print(f"文件 {csv_file_path} 不是UTF-8编码，跳过处理")
    except Exception as e:
        print(f"处理文件 {csv_file_path} 时出错: {e}")


def process_all_csv_files(data_dir, host, user, password, database):
    """处理data目录下的所有CSV文件"""
    try:
        # 连接MySQL数据库
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",  # 明确指定字符集
            use_unicode=True,  # 启用Unicode支持
            autocommit=False,  # 禁用自动提交，手动控制事务
        )
        cursor = conn.cursor()

        # 创建表
        create_horse_table(cursor)

        data_path = Path(data_dir)
        csv_files = data_path.rglob("*.csv")

        processed_count = 0
        skipped_count = 0

        for csv_file in csv_files:
            try:
                print(f"正在处理文件: {csv_file}")
                insert_horses_from_csv(cursor, str(csv_file))
                processed_count += 1
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                skipped_count += 1

        # 提交事务
        conn.commit()
        cursor.close()
        conn.close()

        print(f"处理完成！共处理了 {processed_count} 个CSV文件，跳过了 {skipped_count} 个文件")

    except pymysql.Error as e:
        print(f"数据库连接错误: {e}")


# 主程序
if __name__ == "__main__":
    # 设置路径和数据库连接参数
    DATA_DIR = "d:\\code\\DataScienceIntro\\大作业\\data"

    # MySQL数据库配置
    DB_HOST = "localhost"  # 数据库主机地址
    DB_USER = "root"  # 数据库用户名
    DB_PASSWORD = "tyj810915mysql"  # 数据库密码
    DB_NAME = "horse_race_db"  # 数据库名称

    # 处理所有CSV文件
    process_all_csv_files(DATA_DIR, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
