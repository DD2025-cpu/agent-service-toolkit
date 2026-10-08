# 统一项目参数模型

`ProjectParameters` 是 OilWind-Agent 所有经济计算、模型预测、SHAP 和情景工具的统一外部输入。所有货币字段均使用人民币（CNY），比例均使用小数，例如 8% 写作 `0.08`。

## 项目输入

| 字段 | 单位 | 范围 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `installed_capacity_mw` | MW | `(0, 10000]` | 必填 | 装机容量 |
| `total_investment_yuan` | 元 | `(0, 1e12]` | 二选一 | 与单位投资至少提供一个 |
| `unit_capex_yuan_per_kw` | 元/kW | `(0, 100000]` | 二选一 | 与总投资至少提供一个 |
| `full_load_hours_per_year` | 小时/年 | `(0, 8760]` | 必填 | 年等效满负荷利用小时 |
| `tariff_yuan_per_kwh` | 元/kWh | `[0, 10]` | 必填 | 上网电价 |
| `opex_ratio` | 比例 | `[0, 1]` | 必填 | 年运维成本占总投资比例 |
| `project_id` | 文本 | 1–100字符 | `null` | 项目标识 |
| `province` | 文本 | 1–100字符 | `null` | 项目所在地 |

同时提供总投资和单位投资时，两者必须满足：

```text
总投资（元） = 装机容量（MW） × 1000 × 单位投资（元/kW）
```

## 公开经济假设

| 字段 | 默认值 |
| --- | ---: |
| `currency` | `CNY` |
| `project_life_years` | 20年 |
| `discount_rate` | 0.08 |
| `annual_degradation_rate` | 0.005 |
| `residual_value_rate` | 0.05 |
| `corporate_income_tax_rate` | 0.20 |
| `other_revenue_tax_rate` | 0.0 |
| `carbon_price_yuan_per_ton` | 0元/吨 |
| `emission_reduction_t_per_mwh` | 0.50吨/MWh |

## 现有IRR模型适配

`adapt_project_to_irr_features` 将统一输入转换为已保存 XGBoost v2 模型的五个特征，并严格维持训练顺序：

1. `capacity_mw`
2. `investment_million_cny`
3. `full_hours`
4. `tariff_cny_per_kwh`
5. `opex_ratio`

当前 v2 模型使用已经完成固定汇率换算的人民币口径代理数据集训练，因此适配器只转换字段名称，并将总投资从“元”缩放为“百万元”；在线预测阶段不再进行人民币与越南盾换算。该训练集源于越南公开风电项目，不能表述为真实中国项目样本。

## 参数校验工具

LangGraph 工具名为 `validate_project_parameters`。它接受不完整的 `project` 对象，并返回：

- `valid`：参数是否完整有效；
- `can_calculate`：是否允许进入经济计算；
- `next_action`：继续计算或向用户追问；
- `missing_required`：缺失的必填字段；
- `invalid_parameters`：类型、范围或字段冲突；
- `unit_errors`：旧字段、非人民币字段或疑似单位错误；
- `defaults_used`：本次将采用且用户没有显式提供的默认假设；
- `normalized_parameters`：校验通过后的完整参数；
- `follow_up_questions`：Agent可直接向用户提出的问题。

经济计算、模型预测和情景模拟必须先调用 `require_valid_project_parameters`。校验失败时该守卫抛出 `ProjectParameterValidationError`，从函数边界阻止计算。
