import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
df = pd.read_csv("first_model_results/dl_subject_ranking_first_model_results.csv")

# 创建pictures文件夹（如果不存在）
if not os.path.exists("pictures"):
    os.makedirs("pictures")

# 1. R² Score 排名条形图
plt.figure(figsize=(12, 8))
sorted_df_r2 = df.sort_values("r2", ascending=True)
bars1 = plt.barh(sorted_df_r2["subject"], sorted_df_r2["r2"], color="skyblue")
plt.xlabel("R2 Score")
plt.title("各学科 R2 Score 排名")
plt.grid(axis="x", alpha=0.3)

# 在条形图上添加数值标签
for i, (bar, value) in enumerate(zip(bars1, sorted_df_r2["r2"])):
    plt.text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("pictures/r2_score_ranking.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 2. MAE 对比条形图
plt.figure(figsize=(12, 8))
sorted_df_mae = df.sort_values("mae", ascending=False)
bars2 = plt.bar(sorted_df_mae["subject"], sorted_df_mae["mae"], color="lightcoral")
plt.ylabel("MAE")
plt.title("各学科 MAE 对比")
plt.xticks(rotation=90)
plt.grid(axis="y", alpha=0.3)

# 标注最大值
for i, (bar, value) in enumerate(zip(bars2, sorted_df_mae["mae"])):
    plt.text(
        i, value + max(sorted_df_mae["mae"]) * 0.01, f"{value:.1f}", ha="center", va="bottom", fontsize=8
    )

plt.tight_layout()
plt.savefig("pictures/mae_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 3. MAPE 对比条形图
plt.figure(figsize=(12, 8))
sorted_df_mape = df.sort_values("mape", ascending=False)
norm_mape = plt.Normalize(vmin=sorted_df_mape["mape"].min(), vmax=sorted_df_mape["mape"].max())
colors = plt.cm.Reds(norm_mape(sorted_df_mape["mape"]))
bars3 = plt.bar(sorted_df_mape["subject"], sorted_df_mape["mape"], color=colors)
plt.ylabel("MAPE (%)")
plt.title("各学科 MAPE 对比")
plt.xticks(rotation=90)
plt.grid(axis="y", alpha=0.3)

# 添加数值标签
for i, (bar, value) in enumerate(zip(bars3, sorted_df_mape["mape"])):
    plt.text(
        i, value + max(sorted_df_mape["mape"]) * 0.01, f"{value:.1f}%", ha="center", va="bottom", fontsize=8
    )

plt.tight_layout()
plt.savefig("pictures/mape_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 4. MSE 散点图（按学科）
plt.figure(figsize=(12, 8))
scatter = plt.scatter(range(len(df)), df["mse"], c=df["r2"], cmap="viridis", s=100)
plt.xlabel("学科索引")
plt.ylabel("MSE")
plt.title("各学科 MSE 分布 (颜色表示 R2)")
plt.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label("R2 Score")

# 添加学科标签
for i, subject in enumerate(df["subject"]):
    if df["mse"].iloc[i] > df["mse"].quantile(0.8):  # 只标注高MSE的学科
        plt.annotate(
            subject.split()[0], (i, df["mse"].iloc[i]), xytext=(5, 5), textcoords="offset points", fontsize=7
        )

plt.tight_layout()
plt.savefig("pictures/mse_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

