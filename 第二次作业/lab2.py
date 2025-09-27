# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
df = pd.read_csv("data/train.csv")

# 查看数据基本信息
print("数据集基本信息：")
print(df.info())

# 查看数据描述性统计
print("\n描述性统计：")
print(df.describe())

# 检测缺失值
print("缺失值统计：")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# 处理缺失值
# 对于数值型特征，使用中位数填充
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# 对于分类特征，标记为"Unknown"
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna("Unknown", inplace=True)


# 选择数值型列进行异常值检测，使用的异常值检测方法是 IQR法
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col} 的异常值数量: {len(outliers)}")


# 只计算数值型变量的相关性
correlation_matrix = df[numeric_cols].corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, square=True)
plt.title("特征之间的相关性热力图")
plt.show()

# 找出所有 |r| > 0.5 的特征对
strong_pairs = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col1, col2 = numeric_cols[i], numeric_cols[j]
        corr = correlation_matrix.loc[col1, col2]
        if abs(corr) > 0.5:
            strong_pairs.append((col1, col2, corr))

# 显示结果
print("强相关特征对（|r| > 0.5）：")
for pair in strong_pairs:
    print(f"{pair[0]} 与 {pair[1]}: {pair[2]:.3f}")


# 创建标准化器
scaler = StandardScaler()

# 对 price 进行标准化（z-score 标准化）
df["price_scaled"] = scaler.fit_transform(df[["SalePrice"]])

print("标准化后的 price 前几项：")
print(df["price_scaled"].head())

# 等宽分桶（Equal-width binning）
bins = [0, 150000, 300000, np.inf]
labels = ["低价（0-150K）", "中价（150K-300K）", "高价（300K以上）"]
df["price_level"] = pd.cut(df["SalePrice"], bins=bins, labels=labels)

print("价格等级分布：")
print(df["price_level"].value_counts())


# 获取 price 与其他特征的相关性排序
price_corr = correlation_matrix["SalePrice"].abs().sort_values(ascending=False)
top_3_features = price_corr[1:4]  # 排除自己

print("与 price 相关性最高的三个特征：")
print(top_3_features)
