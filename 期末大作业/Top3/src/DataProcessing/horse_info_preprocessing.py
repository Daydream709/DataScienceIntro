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

# SQL 查询保持不变
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

# 备份结算列
df["raw_win_odds"] = df["win_odds"].fillna(df["win_odds"].median())
df["actual_rank"] = df["finish_rank"]

# 2. 基础预处理
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

# 4. 历史特征计算 (针对前三名优化)
df = df.sort_values(by=["horse_name", "race_id"])

def calculate_history(group):
    group["history_race_count"] = group.reset_index().index
    
    # 历史独赢统计
    is_win = (group["finish_rank"] == 1).astype(int)
    group["history_win_count"] = is_win.shift(1).fillna(0).expanding().sum()
    
    # 历史前三统计 (对本任务最重要)
    is_top3 = (group["finish_rank"] <= 3).astype(int)
    group["history_top3_count"] = is_top3.shift(1).fillna(0).expanding().sum()

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

# --- 目标变量修改：预测是否进入前三名 ---
df["rank_label"] = (df["finish_rank"] <= 3).astype(int)

# --------------------------
# 特征标准化与独热编码
# --------------------------
REAL_NUMERIC = [
    "racing_age", "gate_no", "jockey_weight", "popularity", "horse_weight_cleaned",
    "distance", "pop_vs_mean", "weight_vs_mean", "pop_rank", "history_race_count",
    "history_win_count", "history_top3_count", "history_avg_rank", "history_win_rate",
    "history_top3_rate", "win_rate_vs_race_avg", "top3_rate_vs_race_avg", "weight_per_dist",
]

print("独热编码...")
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()
df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=int)

# 7. 时序切分
unique_races = sorted(df["race_id"].unique())
split_point = int(len(unique_races) * 0.8)
train_data = df[df["race_id"].isin(unique_races[:split_point])].reset_index(drop=True)
test_data = df[df["race_id"].isin(unique_races[split_point:])].reset_index(drop=True)

# 8. 标准化
sc = StandardScaler()
def finalize_X(data, scaler, fit=False):
    num_part = pd.DataFrame(
        scaler.fit_transform(data[REAL_NUMERIC]) if fit else scaler.transform(data[REAL_NUMERIC]),
        columns=REAL_NUMERIC
    )
    cat_part = data.filter(regex="gender_|venue_type_|direction_|weather_|track_status_").reset_index(drop=True)
    return pd.concat([num_part, cat_part], axis=1)

X_train_final = finalize_X(train_data, sc, fit=True)
X_test_final = finalize_X(test_data, sc, fit=False)

# 9. 补回结算列
X_test_final["raw_win_odds"] = test_data["raw_win_odds"].values
X_test_final["actual_rank"] = test_data["actual_rank"].values
X_test_final["race_id"] = test_data["race_id"].values

# 保存结果
os.makedirs("../../data", exist_ok=True)
X_train_final.to_csv("../../data/X_train_final_top3.csv", index=False)
X_test_final.to_csv("../../data/X_test_final_top3.csv", index=False)
train_data["rank_label"].to_csv("../../data/y_train_final_top3.csv", index=False)
test_data["rank_label"].to_csv("../../data/y_test_final_top3.csv", index=False)

print(f"\n✅ 预处理完成 (Top 3 模式)")
print(f"训练集正样本比例: {train_data['rank_label'].mean():.2%}")