from langchain_core.messages import AIMessage, HumanMessage, ToolCall

from oilwind_agent.agent.parameter_extraction import (
    enrich_validation_tool_calls,
    extract_explicit_project_parameters,
)

SCREENSHOT_PROMPT = """这是一个虚构的新项目：装机容量80MW，单位投资5800元/kW，
年利用小时2800小时，上网电价0.42元/kWh，OPEX率0.025。

请同时计算现金流公式IRR和XGBoost模型预测IRR。"""


def test_extracts_all_explicit_parameters_from_screenshot_prompt():
    assert extract_explicit_project_parameters(SCREENSHOT_PROMPT) == {
        "installed_capacity_mw": 80.0,
        "unit_capex_yuan_per_kw": 5800.0,
        "full_load_hours_per_year": 2800.0,
        "tariff_yuan_per_kwh": 0.42,
        "opex_ratio": 0.025,
    }


def test_recovers_fields_when_model_only_extracts_truncated_project_id():
    response = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="validate_project_parameters",
                args={"project": {"project_id": "虚"}},
                id="validation",
            )
        ],
    )

    enriched = enrich_validation_tool_calls(response, [HumanMessage(content=SCREENSHOT_PROMPT)])

    assert enriched.tool_calls[0]["args"]["project"] == {
        "installed_capacity_mw": 80.0,
        "unit_capex_yuan_per_kw": 5800.0,
        "full_load_hours_per_year": 2800.0,
        "tariff_yuan_per_kwh": 0.42,
        "opex_ratio": 0.025,
    }


def test_preserves_invalid_units_for_validator_instead_of_silent_conversion():
    prompt = (
        "装机容量50000kW，单位投资5200元/kW，年利用小时2400小时，"
        "电价1500元/kWh，OPEX率填3，请计算IRR。"
    )

    assert extract_explicit_project_parameters(prompt) == {
        "capacity_kw": 50000.0,
        "unit_capex_yuan_per_kw": 5200.0,
        "full_load_hours_per_year": 2400.0,
        "tariff_yuan_per_kwh": 1500.0,
        "opex_ratio": 3.0,
    }


def test_explicit_percent_is_converted_to_decimal_ratio():
    assert extract_explicit_project_parameters("年运维成本率3%") == {"opex_ratio": 0.03}


def test_does_not_overwrite_complete_projects_in_comparison():
    prompt = (
        "项目A：装机容量50MW、单位投资5200元/kW、年利用小时2400小时、"
        "上网电价0.35元/kWh、OPEX率0.03。项目B：装机容量80MW、"
        "单位投资5800元/kW、年利用小时2800小时、上网电价0.42元/kWh、"
        "OPEX率0.025。"
    )
    project_a = {
        "project_id": "项目A",
        "installed_capacity_mw": 50,
        "unit_capex_yuan_per_kw": 5200,
        "full_load_hours_per_year": 2400,
        "tariff_yuan_per_kwh": 0.35,
        "opex_ratio": 0.03,
    }
    project_b = {
        "project_id": "项目B",
        "installed_capacity_mw": 80,
        "unit_capex_yuan_per_kw": 5800,
        "full_load_hours_per_year": 2800,
        "tariff_yuan_per_kwh": 0.42,
        "opex_ratio": 0.025,
    }
    response = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="validate_project_parameters",
                args={"project": project_a},
                id="validate-a",
            ),
            ToolCall(
                name="validate_project_parameters",
                args={"project": project_b},
                id="validate-b",
            ),
        ],
    )

    enriched = enrich_validation_tool_calls(response, [HumanMessage(content=prompt)])

    assert enriched.tool_calls[0]["args"]["project"] == project_a
    assert enriched.tool_calls[1]["args"]["project"] == project_b


def test_preserves_full_explicit_report_project_name():
    prompt = (
        "请对虚构的碳收益研究项目D做完整评价并保存报告。"
        "装机容量60MW，单位投资6000元/kW，年利用小时2600小时，"
        "上网电价0.38元/kWh，OPEX率0.028。"
    )
    response = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="validate_project_parameters",
                args={
                    "project": {
                        "project_id": "D",
                        "installed_capacity_mw": 60,
                        "unit_capex_yuan_per_kw": 6000,
                        "full_load_hours_per_year": 2600,
                        "tariff_yuan_per_kwh": 0.38,
                        "opex_ratio": 0.028,
                    }
                },
                id="validate-report",
            )
        ],
    )

    enriched = enrich_validation_tool_calls(response, [HumanMessage(content=prompt)])

    assert enriched.tool_calls[0]["args"]["project"]["project_id"] == "碳收益研究项目D"
