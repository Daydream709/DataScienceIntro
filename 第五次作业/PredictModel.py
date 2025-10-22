import pandas as pd
import numpy as np
import os
import joblib  # 用于保存模型
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端避免Tkinter线程问题
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体支持
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# 创建模型结果保存目录
if not os.path.exists("model_results"):
    os.makedirs("model_results")

# 创建模型保存目录
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")


def load_and_preprocess_data(subject_name):
    """
    加载并预处理指定学科的数据
    """
    file_path = os.path.join("download", f"{subject_name}.csv")
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        return None

    # 尝试不同的编码格式读取数据
    try:
        # 首先尝试UTF-8编码
        df = pd.read_csv(file_path, skiprows=1, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            # 如果UTF-8失败，尝试ISO-8859-1编码
            df = pd.read_csv(file_path, skiprows=1, encoding="ISO-8859-1")
        except UnicodeDecodeError:
            # 如果都失败，让pandas自动检测编码
            df = pd.read_csv(file_path, skiprows=1, encoding="utf-8-sig")

    # 清理列名
    df.columns = ["Rank", "Institution", "Country", "Documents", "Cites", "CitesPerPaper", "TopPapers"]

    # 清理数据
    df = df.dropna()

    # 将排名转换为数值型
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")

    # 处理数值列，确保正确转换
    df["Documents"] = pd.to_numeric(df["Documents"].astype(str).str.replace(",", ""), errors="coerce")
    df["Cites"] = pd.to_numeric(df["Cites"].astype(str).str.replace(",", ""), errors="coerce")
    df["CitesPerPaper"] = pd.to_numeric(df["CitesPerPaper"], errors="coerce")
    df["TopPapers"] = pd.to_numeric(df["TopPapers"].astype(str).str.replace(",", ""), errors="coerce")

    # 删除空值
    df = df.dropna()

    # 检查数据质量
    print(f"数据质量检查 - {subject_name}:")
    print(f"  数据量: {len(df)}")
    print(f"  排名范围: {df['Rank'].min()} - {df['Rank'].max()}")
    print(f"  文章数范围: {df['Documents'].min()} - {df['Documents'].max()}")
    print(f"  引用次数范围: {df['Cites'].min()} - {df['Cites'].max()}")

    return df


def create_ranking_model(subject_name):
    """
    为指定学科创建排名预测模型
    """
    print(f"\n正在处理学科: {subject_name}")

    # 加载数据
    df = load_and_preprocess_data(subject_name)
    if df is None or len(df) < 10:
        print(f"数据不足，跳过学科: {subject_name}")
        return None

    print(f"数据量: {len(df)} 条记录")

    # 特征选择
    features = ["Documents", "Cites", "CitesPerPaper", "TopPapers"]
    X = df[features]
    y = df["Rank"]

    # 打乱数据并分割成训练集(60%)、验证集(20%)和测试集(20%)
    # 先将数据打乱
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X_shuffled = df_shuffled[features]
    y_shuffled = df_shuffled["Rank"]

    # 分割数据
    X_temp, X_test, y_temp, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of total
    )

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}, 测试集大小: {len(X_test)}")

    # 数据标准化（使用RobustScaler处理异常值）
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 保存标准化器，以便将来使用模型时进行相同的标准化
    joblib.dump(scaler, f"saved_models/{subject_name}_scaler.pkl")

    # 创建并训练模型
    models = {
        "Linear Regression": {"model": LinearRegression(), "params": {}},
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [None, 10], "min_samples_split": [2, 5]},
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        },
    }

    results = {}
    best_models = {}

    for name, model_info in models.items():
        try:
            model = model_info["model"]
            params = model_info["params"]

            # 如果有参数需要调优，则使用GridSearchCV
            if params:
                # 对于线性回归，我们仍然使用标准化数据
                if name == "Linear Regression":
                    grid_search = GridSearchCV(model, params, cv=3, scoring="r2", n_jobs=-1)
                    grid_search.fit(X_train_scaled, y_train)
                else:
                    grid_search = GridSearchCV(model, params, cv=3, scoring="r2", n_jobs=-1)
                    grid_search.fit(X_train, y_train)

                # 获取最佳模型
                best_model = grid_search.best_estimator_
                best_models[name] = best_model

                # 在验证集上评估
                if name == "Linear Regression":
                    y_val_pred = best_model.predict(X_val_scaled)
                else:
                    y_val_pred = best_model.predict(X_val)
            else:
                # 训练模型
                if name == "Linear Regression":
                    model.fit(X_train_scaled, y_train)
                    y_val_pred = model.predict(X_val_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_val_pred = model.predict(X_val)
                best_models[name] = model

            # 在验证集上计算评估指标
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            results[name] = {
                "model": best_models[name],
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_r2": val_r2,
            }

            print(f"{name} - 验证集 MSE: {val_mse:.2f}, MAE: {val_mae:.2f}, R2: {val_r2:.4f}")
        except Exception as e:
            print(f"{name} 模型训练出错: {e}")
            continue

    # 选择最佳模型（根据验证集R2分数）
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k]["val_r2"])
        best_result = results[best_model_name]
        best_model = best_result["model"]

        print(f"最佳模型: {best_model_name}")

        # 保存最佳模型
        joblib.dump(best_model, f"saved_models/{subject_name}_best_model.pkl")
        print(f"模型已保存至: saved_models/{subject_name}_best_model.pkl")

        # 在测试集上评估最终模型
        if best_model_name == "Linear Regression":
            y_test_pred = best_model.predict(X_test_scaled)
        else:
            y_test_pred = best_model.predict(X_test)

        # 计算测试集评估指标
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # 分析预测结果
        pred_mean = np.mean(y_test_pred)
        pred_std = np.std(y_test_pred)
        actual_mean = np.mean(y_test)
        actual_std = np.std(y_test)

        print(f"预测值统计: 均值={pred_mean:.2f}, 标准差={pred_std:.2f}")
        print(f"实际值统计: 均值={actual_mean:.2f}, 标准差={actual_std:.2f}")

        # 可视化结果
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.6, color="blue")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        plt.xlabel("实际排名")
        plt.ylabel("预测排名")
        plt.title(f"{subject_name} 学科排名预测结果 (最佳模型: {best_model_name})\nR2={test_r2:.4f}")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"model_results/{subject_name}_prediction.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 特征重要性分析（仅对树模型）
        if hasattr(best_model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {"feature": features, "importance": best_model.feature_importances_}
            ).sort_values("importance", ascending=False)

            print("特征重要性:")
            print(feature_importance)

            # 可视化特征重要性
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance["feature"], feature_importance["importance"])
            plt.xlabel("重要性")
            plt.title(f"{subject_name} - 特征重要性 ({best_model_name})")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"model_results/{subject_name}_feature_importance.png", dpi=300, bbox_inches="tight")
            plt.close()

        return {
            "subject": subject_name,
            "best_model": best_model_name,
            "mse": test_mse,
            "mae": test_mae,
            "r2": test_r2,
        }
    else:
        print("所有模型训练失败")
        return None


def main():
    """
    主函数：对所有学科建立排名预测模型
    """
    print("开始建立各学科排名预测模型...")

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir("download") if f.endswith(".csv")]
    subjects = [f.replace(".csv", "") for f in csv_files]

    print(f"共找到 {len(subjects)} 个学科数据文件")

    # 存储所有结果
    all_results = []

    # 对每个学科建立模型
    for subject in subjects:
        try:
            result = create_ranking_model(subject)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"处理学科 {subject} 时出错: {e}")
            continue

    # 汇总结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("r2", ascending=False)

        print("\n=== 各学科模型性能汇总 ===")
        print(results_df[["subject", "best_model", "r2", "mae"]])

        # 保存结果到CSV文件
        results_df.to_csv(
            "model_results/subject_ranking_model_results.csv", index=False, encoding="utf-8-sig"
        )

        # 可视化模型性能
        plt.figure(figsize=(12, 8))
        plt.scatter(range(len(results_df)), results_df["r2"])
        plt.xlabel("学科索引")
        plt.ylabel("R2 分数")
        plt.title("各学科排名预测模型性能 (R2 分数)")
        plt.xticks(range(len(results_df)), results_df["subject"], rotation=90)
        plt.tight_layout()
        plt.savefig("model_results/model_performance_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("\n模型建立完成，结果已保存到 model_results 目录")
        print("训练好的模型已保存到 saved_models 目录")
    else:
        print("没有成功建立任何模型")


if __name__ == "__main__":
    main()
