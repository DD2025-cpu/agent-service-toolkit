# 风电IRR预测Python基线记录

记录日期：2026-08-26

## 1. 项目信息

项目位置：

E:\F_WorkSpace\Wind_IRR_Python_Baseline

基线程序：

src\baseline_pipeline.py

运行状态：

程序完整运行成功，退出代码为0。

## 2. 运行环境

Python版本：3.10.19

XGBoost版本：3.1.2

scikit-learn版本：1.7.2

pandas版本：2.3.3

NumPy版本：2.2.6

Conda环境：windirr

## 3. 数据规模

公开项目数据：541条

数据增强数量：1000条

增强后总样本数：1541条

清洗后可用样本数：552条

参与IRR计算的样本数：552条

成功计算IRR的样本数：542条

IRR计算失败或无效：10条

## 4. 模型输入

特征顺序：

1. capacity_mw
2. investment_billion_vnd
3. full_hours
4. tariff_vnd_per_kwh
5. opex_ratio

预测目标：

irr

类别特征：

当前基线模型未使用类别特征。

编码方式：

当前基线模型未使用类别编码。

## 5. 特征单位

capacity_mw：兆瓦，MW

investment_billion_vnd：十亿越南盾，即10^9 VND

full_hours：年等效利用小时，小时/年

tariff_vnd_per_kwh：上网电价，越南盾/千瓦时

opex_ratio：运维成本占收入的比例，使用小数表示

irr：内部收益率，使用小数表示，例如0.10表示10%

## 6. 数据划分

划分方法：

train_test_split

测试集比例：

test_size=0.2

数据划分随机种子：

random_state=42

## 7. XGBoost基线参数

n_estimators=300

learning_rate=0.05

max_depth=4

subsample=0.8

colsample_bytree=0.8

random_state=42

## 8. 基线评测结果

MAE：0.012824

R²：0.865878

RMSE：当前未找到，待后续增加评测代码后核实。

## 9. 模型保存情况

模型是否保存：否

模型文件位置：未找到

模型加载代码：未找到

当前模型只能通过重新运行程序进行训练。

## 10. 当前基线说明

当前基线完成了：

1. 读取越南公开风电项目数据；
2. 生成增强数据；
3. 计算模拟经济参数和IRR；
4. 训练XGBoost回归模型；
5. 计算MAE和R²；
6. 对比线性回归、随机森林和XGBoost。

当前基线尚未包含：

1. RMSE评测；
2. 模型保存与加载；
3. NPV计算；
4. 投资回收期计算；
5. LCOE计算；
6. SHAP解释；
7. 完整碳价情景；
8. 人民币单位转换；
9. 真实长庆油田项目数据。