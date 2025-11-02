import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
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

# 检查并配置GPU使用
print("TensorFlow版本:", tf.__version__)
print("可用的物理设备:")
physical_devices = tf.config.list_physical_devices()
for i, device in enumerate(physical_devices):
    print(f"  [{i}] {device}")

# 配置GPU使用策略
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # 列出所有GPU设备供用户选择
        print("\n检测到以下GPU设备:")
        for i, gpu in enumerate(gpus):
            print(f"  [{i}] {gpu}")

        # 默认选择第一个独立显卡（通常索引为1，0是集显）
        selected_gpu_index = None
        if len(gpus) > 1:
            # 如果有多个GPU，优先选择第二个（通常是独显）
            selected_gpu_index = 1
            print(f"\n默认选择独显进行训练: {gpus[selected_gpu_index]}")
        else:
            # 只有一个GPU，选择它
            selected_gpu_index = 0
            print(f"\n选择唯一的GPU进行训练: {gpus[selected_gpu_index]}")

        # 设置选定GPU的内存增长
        tf.config.experimental.set_memory_growth(gpus[selected_gpu_index], True)

        # 限制TensorFlow只使用选定的GPU
        tf.config.set_visible_devices(gpus[selected_gpu_index], "GPU")

        print("已配置使用指定GPU进行训练")

    except RuntimeError as e:
        print(f"GPU配置出错: {e}")
else:
    print("未检测到GPU设备，将使用CPU进行训练")

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


def create_small_dataset_model(input_dim):
    """
    创建专为小数据集设计的深度学习模型
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
            layers.Dropout(0.4),
            layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(8, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    return model


def noise_augmentation(X, y, noise_factor=0.05, n_copies=2):
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


def create_ranking_model(subject_name):
    """
    为指定学科创建基于深度学习的排名预测模型（专为小数据集优化）
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

    # 准备特征和标签
    X = df_extended[all_features].values
    y = df_extended["Rank"].values

    # 打乱数据
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # 对于小数据集，使用更简单的划分比例：60%训练，20%验证，20%测试
    split_point_1 = int(0.6 * len(X))
    split_point_2 = int(0.8 * len(X))

    X_train, y_train = X[:split_point_1], y[:split_point_1]
    X_val, y_val = X[split_point_1:split_point_2], y[split_point_1:split_point_2]
    X_test, y_test = X[split_point_2:], y[split_point_2:]

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}, 测试集大小: {len(X_test)}")

    # 数据标准化（使用RobustScaler处理异常值）
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 保存标准化器，以便将来使用模型时进行相同的标准化
    joblib.dump(scaler, f"first_saved_models/{subject_name}_scaler.pkl")

    # 数据增强对于小数据集非常重要
    print("执行数据增强...")
    X_train_scaled, y_train = noise_augmentation(X_train_scaled, y_train, noise_factor=0.03, n_copies=3)
    print(f"数据增强后训练集大小: {len(X_train_scaled)}")

    # 创建专为小数据集设计的模型
    model = create_small_dataset_model(X_train_scaled.shape[1])

    # 显示模型结构
    print("模型结构:")
    model.summary()

    # 训练模型 - 针对小数据集调整参数
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.0001)

    # 显示当前使用的设备
    print("正在使用的设备:")
    logical_devices = tf.config.list_logical_devices()
    for device in logical_devices:
        print(f"  {device}")

    # 确保在指定GPU上进行训练
    with tf.device("/GPU:0"):  # 在我们设置的可见GPU上训练
        history = model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=150,  # 增加训练轮数，配合早停机制
            batch_size=min(16, len(X_train_scaled) // 4),  # 动态调整批次大小
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )

    # 保存模型
    model.save(f"first_saved_models/{subject_name}_dl_model.h5")
    print(f"模型已保存至: first_saved_models/{subject_name}_dl_model.h5")

    # 在测试集上评估最终模型
    with tf.device("/GPU:0"):
        y_test_pred = model.predict(X_test_scaled, verbose=0).flatten()

    # 计算测试集评估指标
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"测试集 MSE: {test_mse:.2f}")
    print(f"测试集 MAE: {test_mae:.2f}")
    print(f"测试集 MAPE: {test_mape:.2f}%")
    print(f"测试集 R2: {test_r2:.4f}")

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
