# horse_race_data_preprocessing.py
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import re
import warnings

warnings.filterwarnings("ignore")  # 忽略警告

# 配置数据库连接参数
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "tyj810915mysql"
DB_NAME = "horse_race_db"

# 创建数据库连接
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?charset=utf8mb4")


def load_horse_race_data():
    """从 horse_race 表加载数据"""
    query = "SELECT * FROM horse_race;"
    df = pd.read_sql(query, engine)
    print(f"加载成功！共 {len(df)} 条记录。")
    return df


def preprocess_horse_race(df):
    """
    对 horse_race 数据进行预处理
    """
    print("\n=== 开始数据预处理 ===")

    # 1. 检查基本信息
    print("\n原始数据信息：")
    print(df.info())
    print("\n前5行数据：")
    print(df.head())

    # 2. 检查缺失值
    print("\n缺失值统计：")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    # 3. 处理缺失值
    df["sprint"] = df["sprint"].fillna(0.0)
    df["win_odds"] = df["win_odds"].fillna(0.0)
    df["horse_weight"] = df["horse_weight"].fillna("")
    df["gap"] = df["gap"].fillna("")

    # 4. 类型转换
    df["race_id"] = df["race_id"].astype("int64")
    df["gate_no"] = df["gate_no"].astype("int64")
    df["jockey_weight"] = df["jockey_weight"].astype("float64")
    df["sprint"] = df["sprint"].astype("float64")
    df["win_odds"] = df["win_odds"].astype("float64")
    df["popularity"] = df["popularity"].astype("int64")
    df["finish_rank"] = df["finish_rank"].astype("int64")

    # 5. 清洗 horse_weight：去除括号内内容，只保留主重量
    def clean_horse_weight(weight_str):
        if pd.isna(weight_str) or not isinstance(weight_str, str):
            return None
        # 使用正则提取第一个数字（如 438）
        match = re.search(r"^\d+", weight_str)
        return int(match.group()) if match else None

    df["horse_weight_cleaned"] = df["horse_weight"].apply(clean_horse_weight)
    df["horse_weight_cleaned"] = df["horse_weight_cleaned"].astype("int64", errors="ignore")

    # 6. 过滤 win_odds > 550 的行
    before_filter = len(df)
    df = df[df["win_odds"] <= 550]
    removed_count = before_filter - len(df)
    if removed_count > 0:
        print(f"已移除 {removed_count} 条 win_odds > 550 的记录。")

    # 7. 检查并删除重复行
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"发现 {duplicates} 条重复记录，已删除。")
        df = df.drop_duplicates()

    # # 8. 检查异常值（例如 finish_rank 不合理）
    # invalid_finish_rank = df[df["finish_rank"] < 1].shape[0]
    # if invalid_finish_rank > 0:
    #     print(f"发现 {invalid_finish_rank} 条 finish_rank 小于 1 的记录，已删除。")
    #     df = df[df["finish_rank"] >= 1]

    # 9. 删除原始 horse_weight 列（可选），保留清洗后版本
    df = df.drop(columns=["horse_weight"])

    print("\n=== 预处理完成 ===")
    print(f"最终数据形状：{df.shape}")
    print("\n清洗后数据信息：")
    print(df.info())

    return df


def save_processed_data(df, output_path=None, table_name=None):
    """
    保存处理后的数据
    """
    if output_path:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"数据已保存到：{output_path}")

    if table_name:
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"数据已写入数据库表：{table_name}")


# 主程序
if __name__ == "__main__":
    # 加载数据
    df = load_horse_race_data()

    # 预处理
    processed_df = preprocess_horse_race(df)

    # 保存结果
    output_csv = "processed_horse_race.csv"
    new_table_name = "horse_race_cleaned"

    save_processed_data(processed_df, output_csv, new_table_name)
