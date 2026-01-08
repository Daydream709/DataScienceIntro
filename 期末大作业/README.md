# 赛马预测项目

（本 README 其实就是项目试验报告的超级简化版本）

## 项目简介

该项目使用机器学习方法对赛马比赛结果进行预测，包含单胜（Top1）和前三（Top3）两种预测类型。

## 目录结构

```
├── Top1/              # 单胜预测模块
│   ├── data/          # 数据文件
│   ├── src/           # 源代码
│   └── result/        # 预测结果
├── Top3/              # 前三预测模块
│   ├── data/          # 数据文件
│   ├── src/           # 源代码
│   └── result/        # 预测结果
└── 赛马预测项目实验报告.md  # 项目实验报告
```

## 模型介绍

### 基础模型

- **CatBoost**: 基于梯度提升的决策树模型
- **LightGBM**: 轻量级梯度提升框架
- **XGBoost**: 高性能梯度提升库
- **TabNet**: 基于注意力机制的表格数据模型

### 集成模型

- **Blending**: 模型融合方法，直接平均各模型预测结果
- **Stacking**: 堆叠集成方法，使用元模型结合基础模型预测

## 运行方法

### 1. 模型训练与调优

```bash
# 进入对应模块目录
cd Top1/src/TuningTraining 或 cd Top3/src/TuningTraining

# 运行模型调优脚本
python blending_tuning_final.py
python stacking_tuning_final.py
```

### 2. 准确率测试

```bash
# 进入对应模块目录
cd Top1/src 或 cd Top3/src

# 运行准确率测试脚本
python accuracy_test.py
```

## 结果说明

### 输出文件

- 模型预测结果：`result/*_tuning_result/*.csv`
- 准确率分析报告：`result/hit_rate_analysis/`
- 模型对比图表：`result/hit_rate_analysis/models_accuracy_comparison.png`

### 评估指标

- AUC: 曲线下面积，评估模型排序能力
- F1 分数: 精确率和召回率的调和平均
- LogLoss: 对数损失，评估概率预测准确性
- 准确率: 模型预测结果与实际结果的匹配程度

## 技术栈

- Python 3.8+
- pandas, numpy: 数据处理
- scikit-learn: 机器学习框架
- catboost, lightgbm, xgboost: 梯度提升模型
- pytorch-tabnet: TabNet 模型
- matplotlib, seaborn: 数据可视化

## 实验报告

详细实验结果请参考：[赛马预测项目实验报告.md](赛马预测项目实验报告.md)