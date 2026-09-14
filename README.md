\# HDMap-AI-QA



\## 面向高精地图的空间质量检测与 AI 异常识别系统



HDMap-AI-QA 是一个基于 Python、GeoPandas、Shapely、Scikit-learn、

PyTorch 和 Streamlit 构建的高精地图质量检测实验系统。



项目围绕道路空间数据质量问题，构建了从道路数据生成、异常注入、

属性/几何/拓扑规则检测，到机器学习异常识别、模型比较、

模型部署以及 Web 可视化展示的完整流程。



本项目主要用于验证 GIS 空间数据工程与 AI 异常识别相结合的技术路线。





\---



\## 1. 项目背景



自动驾驶、高精地图和数字地图生产过程中，道路数据质量直接影响

后续导航、路径规划、空间分析及地图更新任务。



道路数据质量问题可能包括：



\- 道路属性缺失

\- 道路类型非法

\- 限速与道路类型不匹配

\- 车道数异常

\- geometry 缺失

\- geometry 重复

\- 道路长度异常

\- 道路端点疑似断裂

\- 多属性组合异常



因此，本项目构建了一套：



规则质量检测 + AI 异常识别



相结合的混合式质量检测流程。





\---



\## 2. 系统架构



道路 CSV 数据



↓  



数据读取与字段校验



↓  



规则质量检测

\- 属性质量检测

\- 几何质量检测

\- 拓扑质量检测



\+



AI 异常识别

\- 特征工程

\- Random Forest

\- 异常概率预测



↓



综合风险判断



↓



质量统计 / 空间地图 / 异常明细



↓



Streamlit Dashboard





\---



\## 3. 核心功能



\### 3.1 道路数据生成



构建上海经纬度范围内的模拟道路数据，包括：



\- road\_id

\- road\_type

\- speed\_limit

\- lane\_num

\- geometry



道路 geometry 使用 LineString 表示。





\### 3.2 异常注入



为验证质量检测能力，人工注入多种异常，包括：



\- missing\_attribute

\- invalid\_road\_type

\- speed\_limit\_conflict

\- lane\_num\_conflict

\- missing\_geometry

\- duplicate\_geometry

\- short\_geometry

\- possible\_topology\_gap





\### 3.3 属性质量检测



根据不同 road\_type 建立道路属性规则，对以下问题进行检测：



\- 必要字段缺失

\- 道路类型非法

\- 限速范围冲突

\- 车道数范围冲突





\### 3.4 几何质量检测



基于 GeoPandas 和 Shapely 实现：



\- geometry 缺失检测

\- 空 geometry 检测

\- invalid geometry 检测

\- duplicate geometry 检测

\- 极短道路检测





\### 3.5 拓扑质量检测



通过提取道路起点和终点，并进行空间距离分析，

识别疑似道路端点断裂问题。



当前实验使用投影坐标进行距离计算。





\---



\## 4. AI 异常识别



\### 4.1 特征工程



机器学习模型使用的主要特征包括：



\- speed\_limit

\- lane\_num

\- road\_length\_m

\- road\_type



road\_type 使用 One-Hot Encoding 转换为数值特征。





\### 4.2 模型实验



项目对以下模型进行了实验：



\- Logistic Regression

\- Random Forest

\- XGBoost

\- Multi-Layer Perceptron (MLP)





\### 4.3 模型结果



在 1000 条模拟道路样本上进行统一训练和测试。



| Model | Accuracy | Precision | Recall | F1 |

| --- | ---: | ---: | ---: | ---: |

| Logistic Regression | 0.920 | 0.917 | 0.611 | 0.733 |

| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 |

| XGBoost | 1.000 | 1.000 | 1.000 | 1.000 |

| MLP | 0.977 | 0.980 | 0.889 | 0.932 |



Random Forest 和 XGBoost 在当前模拟规则型异常数据上取得最高性能。



需要说明的是，当前数据及异常模式均为人工构造，

因此模型性能主要用于验证完整机器学习工程流程，

不代表真实生产高精地图数据上的实际识别能力。





\---



\## 5. 规则检测与 AI 检测的关系



本项目采用 Hybrid QA（混合质量检测）思想。



规则检测适合处理明确、确定性的质量问题，例如：



\- 字段缺失

\- geometry 重复

\- 道路属性范围冲突

\- 拓扑断裂



AI 模型主要根据道路属性和长度特征学习异常模式。



例如，在测试数据中：



\- R0003 的超高限速和极短道路可被 AI 识别

\- R0008 / R0018 的重复 geometry 由规则系统识别

\- R0019 / R0020 的拓扑断裂由空间规则识别



因此，规则检测与 AI 模型具有互补作用。





\---



\## 6. 模型部署



使用 Scikit-learn Pipeline 将：



\- OneHotEncoder

\- Random Forest



封装为统一模型 Pipeline。



通过 joblib 保存训练模型：



models/hdmap\_rf\_pipeline.joblib



部署阶段可直接加载模型，并对新道路执行：



\- predict()

\- predict\_proba()



实现道路异常类别及异常概率预测。





\---



\## 7. Web Dashboard



使用 Streamlit 构建可交互 Web 系统。



主要功能包括：



\- CSV 道路数据上传

\- 属性质量检测

\- 几何质量检测

\- 拓扑质量检测

\- AI 异常概率预测

\- 综合风险道路识别

\- 质量统计

\- 异常结果分类展示

\- CSV 检测结果下载

\- Folium 道路风险地图





\---



\## 8. 地图可视化



使用 Folium 对道路检测结果进行空间展示。



系统将：



\- 正常道路

\- 综合风险道路



以不同样式显示。



点击道路可以查看：



\- Road ID

\- Road Type

\- Rule Anomaly

\- AI Anomaly

\- AI Probability





\---



\## 9. 技术栈



\### GIS / Spatial Data



\- GeoPandas

\- Shapely

\- WKT

\- CRS

\- EPSG:4326

\- EPSG:3857

\- Folium



\### Data Engineering



\- Python

\- Pandas

\- NumPy



\### Machine Learning



\- Scikit-learn

\- Logistic Regression

\- Random Forest

\- XGBoost



\### Deep Learning



\- PyTorch

\- MLP

\- Tensor

\- DataLoader

\- BCEWithLogitsLoss

\- Adam



\### Application



\- Streamlit

\- joblib





\---



\## 10. 项目目录



```text

03\_\_hdmap\_ai\_quality\_system/



├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│   └── hdmap\_rf\_pipeline.joblib

│

├── outputs/

│   ├── model\_comparison.csv

│   ├── mlp\_result.csv

│   ├── mlp\_loss\_history.csv

│   ├── rf\_feature\_importance.csv

│   ├── model\_f1\_comparison.png

│   ├── mlp\_loss\_curve.png

│   └── rf\_feature\_importance.png

│

├── src/

│   ├── data\_generator.py

│   ├── anomaly\_injector.py

│   ├── spatial\_processing.py

│   ├── attribute\_check.py

│   ├── geometry\_check.py

│   ├── topology\_check.py

│   ├── quality\_pipeline.py

│   ├── feature\_engineering.py

│   ├── ml\_data\_generator.py

│   ├── train\_logistic\_regression.py

│   ├── train\_random\_forest.py

│   ├── train\_xgboost.py

│   ├── train\_mlp.py

│   ├── model\_comparison.py

│   ├── train\_deploy\_model.py

│   └── predict\_new\_roads.py

│

├── web\_app/

│   ├── app.py

│   └── map\_visualization.py

│

└── README.md

