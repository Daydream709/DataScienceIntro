import os
import csv
import pymysql
from pathlib import Path
import re


def insert_race_data(cursor, csv_file_path):
    """从CSV文件中读取数据并插入到horse_race表中"""
    # 从文件名提取race_id（去掉扩展名）
    file_name = Path(csv_file_path).stem
    race_id = int(file_name)  # 将文件名转换为整数

    try:
        # 使用UTF-8编码读取文件
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                # 检查着順列是否为数字，如果不是则跳过该行
                rank_str = row["着順"]
                if not rank_str.isdigit():
                    # 如果着順不是数字（如"中"、"除"等），跳过这一行
                    continue

                # 提取字段数据
                horse_name = row["馬名"]
                racing_age = row["性齢"]  # 性齢列包含性别和年龄，这里只取数字部分
                gate_no = int(row["枠番"])
                jockey_weight = float(row["斤量"])
                sprint = float(row["上り"]) if row["上り"] else None
                win_odds = float(row["単勝"]) if row["単勝"] else None
                popularity = int(row["人気"])
                horse_weight = row["馬体重"]
                finish_rank = int(row["着順"])  # 变量名也更新为finish_rank
                finish_time = row["タイム"]
                gap = row["着差"]
                jockey = row["騎手"]

                # 处理racing_age：提取年龄数字部分
                # 例如"牝3" -> 3, "牡4" -> 4
                racing_age_num = int(re.search(r"\d+", racing_age).group()) if racing_age else None

                # 插入数据到horse_race表 - 使用新的列名finish_rank
                cursor.execute(
                    """
                    INSERT INTO horse_race 
                    (race_id, horse_name, racing_age, gate_no, jockey_weight, 
                     sprint, win_odds, popularity, horse_weight, finish_rank, 
                     finish_time, gap, jockey)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        race_id,
                        horse_name,
                        racing_age_num,
                        gate_no,
                        jockey_weight,
                        sprint,
                        win_odds,
                        popularity,
                        horse_weight,
                        finish_rank,  # 参数也更新为finish_rank
                        finish_time,
                        gap,
                        jockey,
                    ),
                )

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

        data_path = Path(data_dir)
        csv_files = data_path.rglob("*.csv")

        processed_count = 0
        total_rows = 0

        for csv_file in csv_files:
            try:
                print(f"正在处理文件: {csv_file}")
                insert_race_data(cursor, str(csv_file))
                processed_count += 1
                # 统计总行数（这里简化处理，实际需要计算每行数量）
                total_rows += 1
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")

        # 提交事务
        conn.commit()
        cursor.close()
        conn.close()

        print(f"处理完成！共处理了 {processed_count} 个CSV文件")
        print(f"总共插入了 {total_rows} 条记录")

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
