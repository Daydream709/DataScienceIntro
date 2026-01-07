import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# 1. 数据库连接
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "tyj810915mysql",
    "database": "horse_race_db",
    "charset": "utf8mb4",
}
engine = create_engine(
    f'mysql+pymysql://{DB_CONFIG["user"]}:{DB_CONFIG["password"]}@{DB_CONFIG["host"]}/{DB_CONFIG["database"]}?charset={DB_CONFIG["charset"]}'
)

SQL_QUERY = """
SELECT 
  hrc.race_id, hrc.horse_name, hrc.racing_age, hrc.gate_no, hrc.jockey_weight,
  hrc.win_odds, hrc.popularity, hrc.finish_rank, hrc.horse_weight_cleaned,
  h.gender, r.venue_type, r.direction, r.distance, r.weather, r.track_status
FROM horse_race_cleaned hrc
LEFT JOIN horse h ON hrc.horse_name = h.horse_name
LEFT JOIN race r ON hrc.race_id = r.race_id;
"""

print("正在从数据库提取原始数据...")
df = pd.read_sql(SQL_QUERY, engine)
df = df.sort_values(by=["race_id", "gate_no"]).reset_index(drop=True)

# --- 核心操作：将 win_odds 转移到辅助列并从特征池中隔离 ---
# 填充缺失值用于结算，并更名为 raw_win_odds
df["raw_win_odds"] = df["win_odds"].fillna(df["win_odds"].median())
df["actual_rank"] = df["finish_rank"]

# 2. 基础预处理 (不再处理特征用的 win_odds)
base_num_cols = ["racing_age", "gate_no", "jockey_weight", "popularity", "horse_weight_cleaned", "distance"]
for col in base_num_cols:
    df[col] = df[col].fillna(df.groupby("race_id")[col].transform("mean")).fillna(df[col].mean())

cat_cols = ["gender", "venue_type", "direction", "weather", "track_status"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# 3. 组内相对特征
def add_relative_features(group):
    group["pop_vs_mean"] = group["popularity"] / (group["popularity"].mean() + 1e-6)
    group["weight_vs_mean"] = group["jockey_weight"] - group["jockey_weight"].mean()
    group["pop_rank"] = group["popularity"].rank(method="min")
    return group


df = df.groupby("race_id", group_keys=False).apply(add_relative_features)

# 4. 历史特征计算 (修正泄露)
df = df.sort_values(by=["horse_name", "race_id"])


def calculate_history(group):
    group["history_race_count"] = group.reset_index().index
    is_win = (group["finish_rank"] == 1).astype(int)
    group["history_win_count"] = is_win.shift(1).fillna(0).expanding().sum()
    is_top3 = (group["finish_rank"] <= 3).astype(int)
    group["history_top3_count"] = is_top3.shift(1).fillna(0).expanding().sum()

    # 历史平均排名初始化为中性值 7.0
    group["history_avg_rank"] = group["finish_rank"].shift(1).expanding().mean().fillna(7.0)

    group["history_win_rate"] = group["history_win_count"] / (group["history_race_count"] + 1e-6)
    group["history_top3_rate"] = group["history_top3_count"] / (group["history_race_count"] + 1e-6)
    return group


print("计算历史特征...")
df = df.groupby("horse_name", group_keys=False).apply(calculate_history)

# 5. 组内竞争特征
df["win_rate_vs_race_avg"] = df.groupby("race_id")["history_win_rate"].transform(lambda x: x - x.mean())
df["top3_rate_vs_race_avg"] = df.groupby("race_id")["history_top3_rate"].transform(lambda x: x - x.mean())

# 6. 交互特征
df["weight_per_dist"] = df["jockey_weight"] / (df["distance"] / 1000.0)

# --------------------------
# 特征定义 (严格无 win_odds)
# --------------------------
REAL_NUMERIC = [
    "racing_age",
    "gate_no",
    "jockey_weight",
    "popularity",
    "horse_weight_cleaned",
    "distance",
    "pop_vs_mean",
    "weight_vs_mean",
    "pop_rank",
    "history_race_count",
    "history_win_count",
    "history_top3_count",
    "history_avg_rank",
    "history_win_rate",
    "history_top3_rate",
    "win_rate_vs_race_avg",
    "top3_rate_vs_race_avg",
    "weight_per_dist",
]

print("独热编码...")
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()
df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=int)

df["rank_label"] = (df["finish_rank"] == 1).astype(int)

# 7. 时序切分
unique_races = sorted(df["race_id"].unique())
split_point = int(len(unique_races) * 0.8)
train_data = df[df["race_id"].isin(unique_races[:split_point])].reset_index(drop=True)
test_data = df[df["race_id"].isin(unique_races[split_point:])].reset_index(drop=True)

# 8. 标准化
sc = StandardScaler()
# 明确不包含 win_odds 的特征列
X_train_final = pd.concat(
    [
        pd.DataFrame(sc.fit_transform(train_data[REAL_NUMERIC]), columns=REAL_NUMERIC),
        train_data.filter(like="gender_").reset_index(drop=True),  # 这里简化了 dummy 选择逻辑
        train_data.filter(like="venue_type_").reset_index(drop=True),
        train_data.filter(like="direction_").reset_index(drop=True),
        train_data.filter(like="weather_").reset_index(drop=True),
        train_data.filter(like="track_status_").reset_index(drop=True),
    ],
    axis=1,
)

X_test_final = pd.concat(
    [
        pd.DataFrame(sc.transform(test_data[REAL_NUMERIC]), columns=REAL_NUMERIC),
        test_data.filter(like="gender_").reset_index(drop=True),
        test_data.filter(like="venue_type_").reset_index(drop=True),
        test_data.filter(like="direction_").reset_index(drop=True),
        test_data.filter(like="weather_").reset_index(drop=True),
        test_data.filter(like="track_status_").reset_index(drop=True),
    ],
    axis=1,
)

# 9. 补回测试集专用结算列
X_test_final["raw_win_odds"] = test_data["raw_win_odds"].values
X_test_final["actual_rank"] = test_data["actual_rank"].values
X_test_final["race_id"] = test_data["race_id"].values

# 保存
os.makedirs("../../data", exist_ok=True)
X_train_final.to_csv("../../data/X_train_final.csv", index=False)
X_test_final.to_csv("../../data/X_test_final.csv", index=False)
train_data["rank_label"].to_csv("../../data/y_train_final.csv", index=False)
test_data["rank_label"].to_csv("../../data/y_test_final.csv", index=False)

print("\n✅ 预处理完成：已删除特征池中的 win_odds，仅保留 raw_win_odds 用于结算。")
