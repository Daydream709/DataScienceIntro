import os
import csv
import pymysql
import re


def connect_to_database():
    """
    连接到MySQL数据库
    """
    try:
        connection = pymysql.connect(
            host="localhost",  # 请根据实际情况修改
            database="horse_race_db",  # 请根据实际情况修改数据库名
            user="root",  # 请根据实际情况修改用户名
            password="tyj810915mysql",  # 请根据实际情况修改密码
            charset="utf8mb4",
        )
        print("成功连接到MySQL数据库")
        return connection
    except Exception as e:
        print(f"连接数据库时出错: {e}")
        return None


def insert_race_data(connection, race_id, venue_type, direction, distance, weather, track_status):
    """
    插入比赛数据到数据库
    """
    try:
        cursor = connection.cursor()

        # 检查记录是否已存在
        check_query = "SELECT COUNT(*) FROM race WHERE race_id = %s"
        cursor.execute(check_query, (race_id,))
        result = cursor.fetchone()

        if result[0] > 0:
            print(f"比赛ID {race_id} 已存在，跳过插入")
            return

        # 插入数据
        insert_query = """
        INSERT INTO race (race_id, venue_type, direction, distance, weather, track_status)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        record = (race_id, venue_type, direction, distance, weather, track_status)
        cursor.execute(insert_query, record)
        connection.commit()
        print(f"成功插入比赛ID: {race_id}")

    except Exception as e:
        print(f"插入数据时出错: {e}")


def extract_race_info_from_csv(file_path):
    """
    从CSV文件中提取比赛信息
    """
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取表头

        # 从表头找到对应列的索引
        try:
            venue_type_idx = headers.index("場地类型")
            direction_idx = headers.index("方向")
            distance_idx = headers.index("距離")
            weather_idx = headers.index("天気")
            track_status_idx = headers.index("場地状況")
        except ValueError as e:
            print(f"CSV文件 {file_path} 缺少必要的列: {e}")
            return None

        # 读取第一行数据（所有行应该具有相同的比赛信息）
        first_row = next(reader, None)
        if first_row is None:
            print(f"CSV文件 {file_path} 为空")
            return None

        # 提取比赛信息
        venue_type = first_row[venue_type_idx] if venue_type_idx < len(first_row) else ""
        direction = first_row[direction_idx] if direction_idx < len(first_row) else ""
        distance_str = first_row[distance_idx] if distance_idx < len(first_row) else "0"
        weather = first_row[weather_idx] if weather_idx < len(first_row) else ""
        track_status = first_row[track_status_idx] if track_status_idx < len(first_row) else ""

        # 处理距离，提取数字部分
        distance_match = re.search(r"(\d+)", distance_str)
        distance = int(distance_match.group(1)) if distance_match else 0

        return {
            "venue_type": venue_type,
            "direction": direction,
            "distance": distance,
            "weather": weather,
            "track_status": track_status,
        }


def process_csv_files(data_folder):
    """
    处理data文件夹中的所有CSV文件
    """
    connection = connect_to_database()
    if not connection:
        return

    try:
        # 遍历data文件夹及其子文件夹
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    print(f"处理文件: {file_path}")

                    # 从文件名提取race_id (去掉.csv后缀)
                    race_id_str = os.path.splitext(file)[0]

                    # 验证race_id是否为数字
                    if not race_id_str.isdigit():
                        print(f"文件名 {file} 不符合race_id格式，跳过")
                        continue

                    # 直接使用字符串作为race_id，不再转换为int
                    race_id = race_id_str

                    # 提取比赛信息
                    race_info = extract_race_info_from_csv(file_path)
                    if race_info:
                        # 插入数据到数据库
                        insert_race_data(
                            connection,
                            race_id,
                            race_info["venue_type"],
                            race_info["direction"],
                            race_info["distance"],
                            race_info["weather"],
                            race_info["track_status"],
                        )

    finally:
        if connection:
            connection.close()
            print("数据库连接已关闭")


def main():
    # 设置data文件夹路径
    data_folder = "data"

    # 检查data文件夹是否存在
    if not os.path.exists(data_folder):
        print(f"文件夹 {data_folder} 不存在")
        return

    # 开始处理CSV文件
    print("开始处理CSV文件并插入数据库...")
    process_csv_files(data_folder)
    print("处理完成！")


if __name__ == "__main__":
    main()
