# horse_race_data_cleaner.py
import pymysql
from pathlib import Path


def clean_horse_race_data(host, user, password, database):
    """
    删除 horse_race 表中 race_id 对应的 race 表中 direction 为 NULL 或空字符串的比赛记录
    """
    try:
        # 连接数据库
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            use_unicode=True,
            autocommit=False,  # 禁用自动提交，手动控制事务
        )
        cursor = conn.cursor()

        print("连接数据库成功。")

        # 第一步：查询将要删除的 race 记录数量（包括 NULL 和空字符串）
        cursor.execute(
            """
            SELECT 
                COUNT(*) AS total,
                SUM(CASE WHEN direction IS NULL THEN 1 ELSE 0 END) AS null_count,
                SUM(CASE WHEN direction = '' THEN 1 ELSE 0 END) AS empty_string_count
            FROM race;
        """
        )
        total, null_count, empty_string_count = cursor.fetchone()
        print(f"总共有 {total} 条 race 记录，其中：")
        print(f"  - direction 为 NULL 的有 {null_count} 条")
        print(f"  - direction 为空字符串的有 {empty_string_count} 条")

        if null_count == 0 and empty_string_count == 0:
            print("没有需要删除的数据。")
            return

        # 第二步：查询将被删除的 horse_race 记录数量
        cursor.execute(
            """
            SELECT COUNT(*) FROM horse_race hr
            INNER JOIN race r ON hr.race_id = r.race_id
            WHERE r.direction IS NULL OR r.direction = '';
        """
        )
        to_delete_count = cursor.fetchone()[0]
        print(f"将删除 {to_delete_count} 条 horse_race 记录。")

        # 第三步：执行删除操作（使用事务）
        cursor.execute(
            """
            DELETE hr FROM horse_race hr
            INNER JOIN race r ON hr.race_id = r.race_id
            WHERE r.direction IS NULL OR r.direction = '';
        """
        )

        # 提交事务
        conn.commit()
        print(f"删除成功！共删除 {to_delete_count} 条 horse_race 记录。")

    except pymysql.Error as e:
        print(f"数据库错误: {e}")
        conn.rollback()  # 回滚事务
    except Exception as e:
        print(f"其他错误: {e}")
        conn.rollback()
    finally:
        if "conn" in locals():
            cursor.close()
            conn.close()
            print("数据库连接已关闭。")


# 主程序
if __name__ == "__main__":
    # 配置数据库连接参数
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASSWORD = "tyj810915mysql"
    DB_NAME = "horse_race_db"

    # 执行清洗
    clean_horse_race_data(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
