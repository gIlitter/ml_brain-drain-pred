本项目实现了一个使用机器学习技术的员工离职预测系统。它结合了生存分析（Cox 比例风险模型）用于风险评分、特征工程、SMOTE 处理类不平衡、GridSearchCV 超参数调优，以及集成模型（LightGBM、XGBoost 和逻辑回归的组合）来预测员工离职。
系统处理包含员工数据的 CSV 文件（例如 Age、MonthlyIncome、OverTime 等），预测离职（二元：0 表示留任，1 表示离职）。它生成预测结果、PDF 格式的可视化图表，以及审计日志。模块化结构将训练、模型保存和预测分离，便于部署或扩展。

关键亮点：

    数据处理：稳健加载、清洗和预处理。
    特征工程：包括异常值上限、派生比率（例如 IncomePerAge）和 Cox 模型的风险分数。
    不平衡处理：SMOTE 上采样。
    超参数调优：针对 n_estimators、max_depth、learning_rate 的网格搜索。
    集成模型：软投票以提高准确性。
    后处理：基于规则调整高风险案例。
    可视化：PNG 图，包括特征重要性、超参数热力图和阈值调优曲线。
    日志记录：带时间戳的详细日志。
    实用工具：自定义日志、时间格式化、数据去重和 MAPE 计算。

本项目适用于 HR 分析，但可适应类似二元分类任务。

先决条件

    Python 版本：3.8 或更高。
    
依赖项：使用 pip install -r requirements.txt 安装。
创建 requirements.txt 并包含以下内容：

    textpandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    imbalanced-learn
    lightgbm
    xgboost
    statsmodels
    joblib
    
硬件：

    标准机器（CPU 足够；XGBoost/LightGBM 可选 GPU 加速）。
    
数据：

    CSV 文件，列一致（例如 Attrition 为目标）。查询中提供样本（train.csv、test.csv）。
    
注意：

    无互联网访问；所有处理本地化。

详细使用说明
1. 设置

克隆/下载项目：下载提供的代码结构。
安装依赖：

    textpip install -r requirements.txt

准备数据：

    将 train.csv 和 test.csv 置于 data/。
    
确保列匹配：

    Attrition（int: 0/1）、Age、BusinessTravel 等。
    test.csv 无 Attrition 列，不会计算评估指标，但仍生成预测。
    数据可截断；代码通过 pandas 处理大文件。


2. 训练模型 (src/train.py)
此脚本执行整个训练工作流。

命令：

    textpython src/train.py
   
逐步发生的事项：

路径设置：
    
    如果缺失，创建 data/fit、log、model 目录。
    
数据加载：
    
    使用 get_data_source（来自 commonUtil.py）读取 train/test CSV。
    
预处理：
    
    删除无关列（EmployeeNumber、Over18、StandardHours）。
    生存分析：在 Age、MonthlyIncome、OverTime 等特征上拟合 Cox PH 模型，计算 RiskScore。
    特征工程：上限异常值，添加 IncomePerAge、IncomePerYear、TenureRatio、LowPayHighOT、RecentHire。
    类别编码：Department、EducationField 等使用 OneHotEncoder。
    数值缩放：StandardScaler。
    交互项：关键数值上的 PolynomialFeatures（degree=2，仅交互）。

平衡：
    
    应用 SMOTE 处理类不平衡。
    
超参数调优：
    
    LightGBM 和 XGBoost 的 GridSearchCV，参数：n_estimators [100,200]、max_depth [3,5]、learning_rate [0.01,0.05]。
    
评分：

    F1（适合不平衡）。

集成训练：
    
    VotingClassifier（软投票）与调优模型 + LogisticRegression。
    
阈值优化：
    
    在验证分割上扫描 0.3-0.7 以获得最佳 F1。
    
预测与后处理：
    
    在测试上预测，应用高风险案例规则。
    
输出：
    
    保存预测到 data/submission_ultimate.csv。
    如果测试有标签，记录准确率/F1。
    在 data/fit/ 生成 png 可视化。

日志：
    
    在 log/train_YYYYMMDD.log 中记录详细步骤、参数、分数。

预期运行时间：
    
    标准硬件上 ~10-30 秒（取决于数据大小；测试于 ~1k 行）。
    
自定义：
    
    在代码中调整参数网格以进行更广泛调优。
    修改 ultimate_features 函数添加新派生特征。

3. 保存模型工件 (src/test.py)
训练后运行此脚本以持久化模型。

命令：

    textpython src/test.py
   
发生的事项：

    将集成模型、预处理器、多项式转换器转储为 PKL 文件到 model/。
    将最佳阈值写入 model/threshold.txt。
    记录确认。
    
注意：

    必须在 train.py 后运行，因为它从那里访问变量（通过 from src.train import * 导入）。
   
4. 在新数据上预测 (src/predict.py)
用于对新/未见数据进行推理。

命令：

    textpython src/predict.py
   
逐步发生的事项：

加载工件：

    从 model/（集成模型、预处理器、多项式、阈值）。
    
数据加载：
    
    读取 data/test.csv（可修改 test_path 用于自定义文件）。
    
预处理：

    镜像训练（删除、工程、编码、缩放、交互）。
    
预测：
  
    计算概率，应用阈值。
    
后处理：
  
    调整高风险规则。
输出：
  
    保存到 data/submission_predict.csv。
    
日志：
  
    在 log/predict_YYYYMMDD.log。

自定义：

    更改脚本中的 test_path 用于不同输入 CSV。
    对于批量预测，在循环或函数中包装。

5. 查看输出

预测：
    
    带有 'Attrition' 列（0/1）的 CSV。
可视化（data/fit/ 中的 png 文件）：

    feature_importance.png：顶部 15 特征的条形图（来自 LightGBM）。
    grid_search_lgb.png / grid_search_xgb.png：参数网格的 F1 分数热力图。
    threshold_tuning.png：F1 vs. 阈值的线图，标记最佳。


日志：

    带时间戳、参数、分数的文本文件。

故障排除

ModuleNotFoundError：
    
    确保依赖安装；检查 Python 路径。
    
FileNotFoundError：
  
    验证 data/ 中的 CSV；目录自动创建。
    
ValueError (列不匹配)：
  
    确保测试数据有与训练相同的特征（Attrition 可选）。
    
低 F1/准确率：
  
    不平衡或小数据；尝试扩展参数网格或添加特征。
    
PDF 生成问题：
  
    确保 matplotlib/seaborn 安装；无显示需要（无头保存）。
    
内存错误：
  
    对于大数据，减少 SMOTE 或使用子集。
    
日志未生成：
    
    检查 util/logUtil.py 权限。

示例结果（来自样本日志）

Cox PH 摘要：

    特征如 OverTime 的风险比率（高风险）。
    
RiskScore 平均：

    训练 ~ -3.91，测试 ~ -3.85。
    
最佳参数：
  
    LGB {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 200}；XGB {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}。
    
最佳阈值：
    
    0.510（验证 F1: 0.9239）。
    
测试指标：
    
    准确率 85.71%，F1 0.5000。

扩展

添加更多模型：

    包含在 VotingClassifier 中。
    
API 部署：

    在 Flask/FastAPI 中包装 predict.py。

Hyperopt/Ray Tune：

    用于高级调优。

特征选择：

    添加 SHAP 用于可解释性。
