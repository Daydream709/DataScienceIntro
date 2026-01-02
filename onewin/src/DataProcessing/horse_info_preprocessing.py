# ======================================
# 赛马数据：最终版特征工程 (单胜预测版)
# ======================================
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

# --- 关键备份：原始结算数据 ---
df["raw_win_odds"] = df["win_odds"].fillna(df["win_odds"].median())
df["actual_rank"] = df["finish_rank"]

# 2. 基础预处理
df["win_odds"] = df["win_odds"].clip(1.0, 500.0)
base_num_cols = [
    "racing_age",
    "gate_no",
    "jockey_weight",
    "win_odds",
    "popularity",
    "horse_weight_cleaned",
    "distance",
]
for col in base_num_cols:
    df[col] = df[col].fillna(df.groupby("race_id")[col].transform("mean")).fillna(df[col].mean())

cat_cols = ["gender", "venue_type", "direction", "weather", "track_status"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# 3. 组内相对特征
def add_relative_features(group):
    group["odds_vs_mean"] = group["win_odds"] / (group["win_odds"].mean() + 1e-6)
    group["pop_vs_mean"] = group["popularity"] / (group["popularity"].mean() + 1e-6)
    group["weight_vs_mean"] = group["jockey_weight"] - group["jockey_weight"].mean()
    group["race_odds_rank"] = group["win_odds"].rank(method="min")
    return group


print("计算组内相对特征...")
df = df.groupby("race_id", group_keys=False).apply(add_relative_features)

# 4. 历史特征计算 (协同调整：侧重单胜)
df = df.sort_values(by=["horse_name", "race_id"])


def calculate_history(group):
    # 历史参赛总数
    group["history_race_count"] = group.reset_index().index

    # 历史冠军次数 (Win)
    is_win = (group["finish_rank"] == 1).astype(int)
    group["history_win_count"] = is_win.shift(1).fillna(0).expanding().sum()

    # 历史前三次数 (Top3 - 作为实力底蕴参考)
    is_top3 = (group["finish_rank"] <= 3).astype(int)
    group["history_top3_count"] = is_top3.shift(1).fillna(0).expanding().sum()

    # 历史平均排名
    group["history_avg_rank"] = (
        group["finish_rank"].shift(1).expanding().mean().fillna(group["finish_rank"].iloc[0])
    )

    # 核心指标：单胜率
    group["history_win_rate"] = group["history_win_count"] / (group["history_race_count"] + 1e-6)
    # 辅助指标：前三入围率
    group["history_top3_rate"] = group["history_top3_count"] / (group["history_race_count"] + 1e-6)

    return group


print("计算马匹历史演化特征 (Win & Top3)...")
df = df.groupby("horse_name", group_keys=False).apply(calculate_history)

# 5. 交互特征
df["win_odds_log"] = np.log1p(df["win_odds"])
df["weight_per_dist"] = df["jockey_weight"] / (df["distance"] / 1000.0)

# --------------------------
# 核心修正 1：更新数值列列表 (加入新历史特征)
# --------------------------
REAL_NUMERIC = [
    "racing_age",
    "gate_no",
    "jockey_weight",
    "win_odds",
    "popularity",
    "horse_weight_cleaned",
    "distance",
    "odds_vs_mean",
    "pop_vs_mean",
    "weight_vs_mean",
    "race_odds_rank",
    "history_race_count",
    "history_win_count",
    "history_top3_count",
    "history_avg_rank",
    "history_win_rate",
    "history_top3_rate",
    "win_odds_log",
    "weight_per_dist",
]

# --------------------------
# 核心修正 2：类别特征独热编码
# --------------------------
print("对类别特征进行独热编码...")
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()

df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=int)

# --------------------------
# 核心修正 3：目标标签改为“单胜” (1st Place Only)
# --------------------------
df["rank_label"] = (df["finish_rank"] == 1).astype(int)

# 6. 时序切分
unique_races = sorted(df["race_id"].unique())
split_point = int(len(unique_races) * 0.8)
train_ids = unique_races[:split_point]
test_ids = unique_races[split_point:]

train_data = df[df["race_id"].isin(train_ids)].reset_index(drop=True)
test_data = df[df["race_id"].isin(test_ids)].reset_index(drop=True)

# 7. 分区标准化逻辑
print("执行分区标准化...")
EXCLUDE_LIST = ["rank_label", "finish_rank", "horse_name", "race_id", "raw_win_odds", "actual_rank"]
dummy_cols = [c for c in df.columns if c not in REAL_NUMERIC and c not in EXCLUDE_LIST]

sc = StandardScaler()

# 训练集
train_numeric_scaled = sc.fit_transform(train_data[REAL_NUMERIC])
X_train_final = pd.concat(
    [pd.DataFrame(train_numeric_scaled, columns=REAL_NUMERIC), train_data[dummy_cols].reset_index(drop=True)],
    axis=1,
)

# 测试集
test_numeric_scaled = sc.transform(test_data[REAL_NUMERIC])
X_test_final = pd.concat(
    [pd.DataFrame(test_numeric_scaled, columns=REAL_NUMERIC), test_data[dummy_cols].reset_index(drop=True)],
    axis=1,
)

# 补回结算列
X_test_final["raw_win_odds"] = test_data["raw_win_odds"].values
X_test_final["actual_rank"] = test_data["actual_rank"].values
X_test_final["race_id"] = test_data["race_id"].values  # <--- 确保加上这一行

# 8. 保存数据 (请确保 ../../data/ 目录已存在)
os.makedirs("../../data", exist_ok=True)
X_train_final.to_csv("../../data/X_train_final.csv", index=False, encoding="utf-8-sig")
X_test_final.to_csv("../../data/X_test_final.csv", index=False, encoding="utf-8-sig")
train_data["rank_label"].to_csv("../../data/y_train_final.csv", index=False)
test_data["rank_label"].to_csv("../../data/y_test_final.csv", index=False)

print("\n" + "=" * 40)
print("✅ 单胜特征工程完成！")
print(f"目标任务: 单胜 (Finish Rank == 1)")
print(f"新增特征: history_win_count, history_win_rate")
print(f"样本比例: 1st Place vs Others 约为 1:{len(X_train_final)//len(train_ids) - 1}")
print("=" * 40)
