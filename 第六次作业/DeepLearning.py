import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体支持
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# 创建模型结果保存目录
if not os.path.exists("first_model_results"):
    os.makedirs("first_model_results")

# 创建模型保存目录
if not os.path.exists("first_saved_models"):
    os.makedirs("first_saved_models")


def mean_absolute_percentage_error(y_true, y_pred):
    """计算MAPE指标"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除零错误
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


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


def create_extended_features(df):
    """
    创建扩展特征
    """
    # 创建衍生特征
    df = df.copy()
    df["CitesPerDocument"] = df["Cites"] / (df["Documents"] + 1e-8)  # 避免除零
    df["TopPapersRatio"] = df["TopPapers"] / (df["Documents"] + 1e-8)
    df["LogDocuments"] = np.log(df["Documents"] + 1)
    df["LogCites"] = np.log(df["Cites"] + 1)
    df["LogTopPapers"] = np.log(df["TopPapers"] + 1)

    return df


def create_optimized_model(input_dim):
    """
    创建优化的深度学习模型
    """
    model = keras.Sequential(
        [
            layers.Dense(
                128, activation="relu", input_shape=(input_dim,), kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    return model


def create_simple_model(input_dim):
    """
    创建简化版深度学习模型（适用于小数据集）
    """
    model = keras.Sequential(
        [
            layers.Dense(
                64, activation="relu", input_shape=(input_dim,), kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    return model


# 新增一个更适合小数据集的模型
def create_tiny_model(input_dim):
    """
    创建极简版深度学习模型（适用于极小数据集）
    """
    model = keras.Sequential(
        [
            layers.Dense(
                32,
                activation="relu",
                input_shape=(input_dim,),
                kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="mse", metrics=["mae"])

    return model


def noise_augmentation(X, y, noise_factor=0.05, n_copies=3):
    """
    添加噪声进行数据增强（适用于小数据集）
    """
    augmented_X = X.copy()
    augmented_y = y.copy()

    for _ in range(n_copies):
        # 添加高斯噪声
        noise_X = X + np.random.normal(0, noise_factor * X.std(axis=0), X.shape)
        noise_y = y + np.random.normal(0, noise_factor * y.std(), y.shape)

        augmented_X = np.vstack([augmented_X, noise_X])
        augmented_y = np.hstack([augmented_y, noise_y])

    return augmented_X, augmented_y


def bagging_ensemble(X_train, y_train, X_val, y_val, X_test, y_test, n_models=5):
    """
    实现Bagging集成方法提升小样本数据集性能
    """
    models = []
    val_predictions = []
    test_predictions = []

    for i in range(n_models):
        # 创建子模型
        model = create_tiny_model(X_train.shape[1])
        models.append(model)

        # Bootstrap采样
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_bootstrap = X_train[indices]
        y_bootstrap = y_train[indices]

        # 训练子模型
        model.fit(
            X_bootstrap,
            y_bootstrap,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=8,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.0001),
            ],
        )

        # 收集验证集和测试集预测结果
        val_pred = model.predict(X_val, verbose=0).flatten()
        test_pred = model.predict(X_test, verbose=0).flatten()
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)

    # 平均所有模型的预测结果
    ensemble_val_pred = np.mean(val_predictions, axis=0)
    ensemble_test_pred = np.mean(test_predictions, axis=0)

    # 计算集成模型性能
    val_mse = mean_squared_error(y_val, ensemble_val_pred)
    test_mse = mean_squared_error(y_test, ensemble_test_pred)

    print(f"集成模型验证集MSE: {val_mse:.2f}")
    print(f"集成模型测试集MSE: {test_mse:.2f}")

    return ensemble_test_pred, models


def cross_validation_training(X, y, subject_name, n_splits=5):
    """
    使用K折交叉验证评估模型性能
    """
    kfold = KFold(n_splits=min(n_splits, len(X) // 5), shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # 创建并训练模型
        model = create_simple_model(X_train_fold.shape[1])

        # 训练模型
        model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=50,
            verbose=0,
            batch_size=16,
        )

        # 评估
        y_pred = model.predict(X_val_fold, verbose=0).flatten()
        mse = mean_squared_error(y_val_fold, y_pred)
        fold_scores.append(mse)

    return np.mean(fold_scores)


def create_ranking_model(subject_name):
    """
    为指定学科创建基于深度学习的排名预测模型
    """
    print(f"\n正在处理学科: {subject_name}")

    # 加载数据
    df = load_and_preprocess_data(subject_name)
    if df is None or len(df) < 10:
        print(f"数据不足，跳过学科: {subject_name}")
        return None

    print(f"数据量: {len(df)} 条记录")

    # 特征工程：创建扩展特征
    df_extended = create_extended_features(df)

    # 特征选择（包含扩展特征）
    base_features = ["Documents", "Cites", "CitesPerPaper", "TopPapers"]
    extended_features = ["CitesPerDocument", "TopPapersRatio", "LogDocuments", "LogCites", "LogTopPapers"]
    all_features = base_features + extended_features

    X = df_extended[all_features]
    y = df_extended["Rank"]

    # 打乱数据并分割成训练集(70%)、验证集(15%)和测试集(15%)
    df_shuffled = df_extended.sample(frac=1, random_state=42).reset_index(drop=True)
    X_shuffled = df_shuffled[all_features]
    y_shuffled = df_shuffled["Rank"]

    # 分割数据
    X_temp, X_test, y_temp, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42  # 约0.15/(1-0.15)
    )

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}, 测试集大小: {len(X_test)}")

    # 数据标准化（使用RobustScaler处理异常值）
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 保存标准化器，以便将来使用模型时进行相同的标准化
    joblib.dump(scaler, f"first_saved_models/{subject_name}_scaler.pkl")

    # 对于非常小的数据集，添加噪声增强
    if len(X_train) < 30:
        print("数据量较小，执行数据增强...")
        X_train_scaled, y_train = noise_augmentation(
            X_train_scaled, y_train.values, noise_factor=0.03, n_copies=2
        )
        print(f"数据增强后训练集大小: {len(X_train_scaled)}")

    # 根据数据量选择模型复杂度
    if len(X_train) > 50:
        model = create_optimized_model(X_train_scaled.shape[1])
        epochs = 100
        batch_size = 32
    elif len(X_train) > 20:
        model = create_simple_model(X_train_scaled.shape[1])
        epochs = 80
        batch_size = 16
    else:
        model = create_tiny_model(X_train_scaled.shape[1])
        epochs = 100
        batch_size = 8

    # 显示模型结构
    print("模型结构:")
    model.summary()

    # 训练模型
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=0.0001)

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    # 保存模型
    model.save(f"first_saved_models/{subject_name}_dl_model.h5")
    print(f"模型已保存至: first_saved_models/{subject_name}_dl_model.h5")

    # 在测试集上评估最终模型
    y_test_pred = model.predict(X_test_scaled, verbose=0).flatten()

    # 对于很小的数据集，尝试集成学习方法
    if len(X_train) < 25 and len(X_test) > 0:
        print("使用集成学习方法提升性能...")
        ensemble_pred, ensemble_models = bagging_ensemble(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, n_models=5
        )
        # 使用集成模型的结果作为最终预测
        y_test_pred = ensemble_pred

    # 计算测试集评估指标
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"测试集 MSE: {test_mse:.2f}")
    print(f"测试集 MAE: {test_mae:.2f}")
    print(f"测试集 MAPE: {test_mape:.2f}%")
    print(f"测试集 R2: {test_r2:.4f}")

    # 分析预测结果
    pred_mean = np.mean(y_test_pred)
    pred_std = np.std(y_test_pred)
    actual_mean = np.mean(y_test)
    actual_std = np.std(y_test)

    print(f"预测值统计: 均值={pred_mean:.2f}, 标准差={pred_std:.2f}")
    print(f"实际值统计: 均值={actual_mean:.2f}, 标准差={actual_std:.2f}")

    # 可视化训练过程
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="训练损失")
    plt.plot(history.history["val_loss"], label="验证损失")
    plt.title("模型损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="训练MAE")
    plt.plot(history.history["val_mae"], label="验证MAE")
    plt.title("模型MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"first_model_results/{subject_name}_training_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("实际排名")
    plt.ylabel("预测排名")
    plt.title(f"{subject_name} 学科排名预测结果\nMSE={test_mse:.2f}, MAPE={test_mape:.2f}%, R2={test_r2:.4f}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"first_model_results/{subject_name}_dl_prediction.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "subject": subject_name,
        "mse": test_mse,
        "mae": test_mae,
        "mape": test_mape,
        "r2": test_r2,
    }


def main():
    """
    主函数：对所有学科建立基于深度学习的排名预测模型
    """
    print("开始建立各学科基于深度学习的排名预测模型...")

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

        print("\n=== 各学科深度学习模型性能汇总 ===")
        print(results_df[["subject", "r2", "mse", "mae", "mape"]])

        # 保存结果到CSV文件
        results_df.to_csv(
            "first_model_results/dl_subject_ranking_first_model_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 可视化模型性能
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = ["r2", "mse", "mae", "mape"]
        titles = ["R2 分数", "MSE", "MAE", "MAPE (%)"]

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            ax.bar(range(len(results_df)), results_df[metric])
            ax.set_xlabel("学科索引")
            ax.set_ylabel(title)
            ax.set_title(f"各学科排名预测模型性能 ({title})")
            ax.set_xticks(range(len(results_df)))
            ax.set_xticklabels(results_df["subject"], rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig("first_model_results/dl_model_performance_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("\n深度学习模型建立完成，结果已保存到 first_model_results 目录")
        print("训练好的模型已保存到 first_saved_models 目录")
    else:
        print("没有成功建立任何模型")


if __name__ == "__main__":
    main()
