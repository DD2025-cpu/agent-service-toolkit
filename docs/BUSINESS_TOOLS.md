# OilWind-Agent业务工具

当前主Agent注册了一个参数校验工具和六个业务工具。所有项目输入均使用`ProjectParameters`人民币口径，所有业务工具在计算前调用同一参数守卫。

## 工具顺序

1. `validate_project_parameters`：检查字段、单位、范围和默认假设。
2. `calculate_economic_metrics`：公式计算IRR、NPV、静态/动态回收期和LCOE。
3. `predict_irr`：加载已保存XGBoost模型预测IRR，不重新训练。
4. `explain_irr_prediction`：使用同一模型工件返回单项目精确TreeSHAP贡献。
5. `simulate_scenarios`：运行悲观、基准、乐观及电价、利用小时、OPEX、碳价和组合情景。
6. `compare_wind_projects`：统一计算并按公式IRR比较两个及以上项目。
7. `generate_wind_project_report`：生成Markdown报告并保存到`outputs/reports`。

## 统一返回协议

每个业务工具返回JSON可序列化对象：

- `schema_version`：当前为1；
- `tool`：稳定工具名；
- `status=success`：计算成功，并携带结构化结果；
- `status=needs_input`：参数不足，`validation`中包含缺失项和追问问题；
- `status=error`：模型工件、情景名称或文件操作失败，`error`中包含类型和消息。

Agent只能在`validate_project_parameters.can_calculate=true`后继续调用业务工具。经济指标、预测、解释、情景和报告函数内部仍会再次校验，防止绕过Agent提示词直接计算。

## 模型工件

模型工件位于`src/oilwind_agent/model/artifacts`。运行时校验：

- 工件结构版本；
- 模型版本`2.0.0`；
- 五个特征的名称和顺序；
- 模型文件SHA-256；
- XGBoost模型内部特征名称。

模型使用越南公开风电项目样本经论文固定汇率换算、人民币字段统一和既定经济口径处理后形成的“人民币口径代理数据集”训练。它用于论文的方法验证、基线比较和预评价，不应表述为基于真实中国风电项目样本训练。公式经济指标和实际项目尽调数据仍是项目经济评价的主要依据。

## 情景口径

情景范围从论文基线等价迁移到人民币电价区间，继续使用随机种子42。每个情景重新使用相同随机种子和相同抽样顺序，以保持不同情景间的抽样分位可比。
