```
├── .venv/                 # 虚拟环境文件夹
├── Top1/                  # 预测马匹获得第一名的相关代码和数据
│   ├── data/              # 预处理后的训练和测试数据
│   ├── result/            # 模型训练和预测结果
│   │   └── ...            # 模型训练结果文件(同Top3)
│   └── src/               # 源代码文件
│       ├── DataProcessing/       # 数据处理相关代码
│       ├── TuningTraining/       # 模型调优和训练代码
│       ├── catboost_info/        # CatBoost训练日志
│       └── accuracy_test.py       # 模型准确率测试和评估
├── Top3/                  # 预测马匹进入前三名的相关代码和数据
│   ├── data/              # 原始数据和预处理后的训练测试数据
│   ├── result/            # 模型训练和预测结果
│   └── src/               # 源代码文件
│       ├── DataProcessing/       # 数据处理相关代码（与Top1相同）
│       ├── GeneralTraining/      # 一般模型训练代码
│       ├── TuningTraining/       # 模型调优和训练代码
│       ├── catboost_info/        # CatBoost训练日志
│       └── accuracy_test.py       # 模型准确率测试和评估
└── README.md              # 项目说明文档
```
