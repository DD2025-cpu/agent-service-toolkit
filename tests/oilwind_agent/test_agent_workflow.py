import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from oilwind_agent.agent.graph import build_oilwind_agent
from oilwind_agent.agent.prompts import OILWIND_SYSTEM_PROMPT

PROJECT = {
    "project_id": "A",
    "installed_capacity_mw": 50,
    "unit_capex_yuan_per_kw": 5_200,
    "full_load_hours_per_year": 2_400,
    "tariff_yuan_per_kwh": 0.35,
    "opex_ratio": 0.03,
}


class FakeToolModel(FakeMessagesListChatModel):
    """A deterministic chat model that accepts LangChain tool binding."""

    def __init__(self, responses: list[BaseMessage]):
        super().__init__(responses=responses)

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[ToolCall(name=name, args=args, id=call_id)],
    )


def tool_names(messages: list[BaseMessage]) -> list[str]:
    return [message.name for message in messages if isinstance(message, ToolMessage)]


async def invoke_with_responses(
    responses: list[BaseMessage],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from oilwind_agent.tools import business

    monkeypatch.setattr(business, "REPORT_DIR", tmp_path)
    model = FakeToolModel(responses)
    graph = build_oilwind_agent()
    with patch("oilwind_agent.agent.graph.get_model", return_value=model):
        return await graph.ainvoke({"messages": [HumanMessage(content="执行验收任务")]})


def tool_payload(message: ToolMessage) -> dict:
    if isinstance(message.artifact, dict):
        return message.artifact
    if isinstance(message.content, str):
        return json.loads(message.content)
    raise TypeError("Tool message did not contain a structured payload")


@pytest.mark.parametrize(
    ("business_tool", "business_args", "expected_field"),
    [
        ("calculate_economic_metrics", {"project": PROJECT}, "metrics"),
        ("predict_irr", {"project": PROJECT}, "predicted_irr"),
        ("explain_irr_prediction", {"project": PROJECT}, "explanation"),
        (
            "simulate_scenarios",
            {"project": PROJECT, "scenario_names": ["pessimistic", "baseline", "optimistic"]},
            "results",
        ),
        (
            "compare_wind_projects",
            {
                "projects": [
                    PROJECT,
                    PROJECT | {"project_id": "B", "tariff_yuan_per_kwh": 0.45},
                ]
            },
            "projects",
        ),
        (
            "generate_wind_project_report",
            {"project": PROJECT, "include_scenarios": True},
            "report_path",
        ),
    ],
    ids=["formula-irr", "model-prediction", "shap", "scenarios", "comparison", "full-report"],
)
@pytest.mark.asyncio
async def test_agent_routes_validated_project_to_business_tool(
    business_tool,
    business_args,
    expected_field,
    monkeypatch,
    tmp_path,
):
    responses = [
        tool_call("validate_project_parameters", {"project": PROJECT}, "validate"),
        tool_call(business_tool, business_args, "business"),
        AIMessage(content="已按工具结果完成。"),
    ]

    result = await invoke_with_responses(responses, monkeypatch, tmp_path)
    messages = result["messages"]

    assert tool_names(messages) == ["validate_project_parameters", business_tool]
    payload = next(
        tool_payload(message)
        for message in messages
        if isinstance(message, ToolMessage) and message.name == business_tool
    )
    assert expected_field in payload


@pytest.mark.asyncio
async def test_missing_parameters_stop_then_continue_on_same_thread(monkeypatch, tmp_path):
    from oilwind_agent.tools import business

    monkeypatch.setattr(business, "REPORT_DIR", tmp_path)
    responses = [
        tool_call(
            "validate_project_parameters",
            {"project": {"installed_capacity_mw": 50}},
            "missing",
        ),
        tool_call("validate_project_parameters", {"project": PROJECT}, "complete"),
        tool_call("calculate_economic_metrics", {"project": PROJECT}, "calculate"),
        AIMessage(content="公式计算已完成。"),
    ]
    model = FakeToolModel(responses)
    graph = build_oilwind_agent(MemorySaver())
    config = {"configurable": {"thread_id": "missing-then-continue"}}

    with patch("oilwind_agent.agent.graph.get_model", return_value=model):
        first = await graph.ainvoke(
            {"messages": [HumanMessage(content="容量50MW，帮我算IRR")]},
            config,
        )
        second = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="补充：单位投资5200元/kW，2400小时，电价0.35元/kWh，OPEX 0.03"
                    )
                ]
            },
            config,
        )

    assert tool_names(first["messages"]) == ["validate_project_parameters"]
    assert "已暂停计算" in first["messages"][-1].content
    assert "上网电价" in first["messages"][-1].content
    assert tool_names(second["messages"])[-2:] == [
        "validate_project_parameters",
        "calculate_economic_metrics",
    ]


@pytest.mark.asyncio
async def test_invalid_unit_and_outlier_stop_before_calculation(monkeypatch, tmp_path):
    invalid_project = PROJECT | {
        "capacity_kw": 50_000,
        "opex_ratio": 3,
        "tariff_yuan_per_kwh": 1_500,
    }
    responses = [
        tool_call("validate_project_parameters", {"project": invalid_project}, "invalid"),
    ]

    result = await invoke_with_responses(responses, monkeypatch, tmp_path)
    messages = result["messages"]

    assert tool_names(messages) == ["validate_project_parameters"]
    assert "已暂停计算" in messages[-1].content
    assert "0.03" in messages[-1].content
    assert "人民币元/kWh" in messages[-1].content


@pytest.mark.asyncio
async def test_tool_error_stops_workflow_without_fabricated_values(monkeypatch, tmp_path):
    responses = [
        tool_call("validate_project_parameters", {"project": PROJECT}, "validate"),
        tool_call(
            "simulate_scenarios",
            {"project": PROJECT, "scenario_names": ["does_not_exist"]},
            "scenario-error",
        ),
    ]

    result = await invoke_with_responses(responses, monkeypatch, tmp_path)
    messages = result["messages"]

    assert tool_names(messages) == ["validate_project_parameters", "simulate_scenarios"]
    assert "业务工具执行失败" in messages[-1].content
    assert "没有生成可用数值结果" in messages[-1].content


@pytest.mark.asyncio
async def test_tool_schema_error_is_caught_without_returning_to_model(monkeypatch, tmp_path):
    responses = [tool_call("calculate_economic_metrics", {}, "bad-schema")]

    result = await invoke_with_responses(responses, monkeypatch, tmp_path)
    messages = result["messages"]

    assert tool_names(messages) == ["calculate_economic_metrics"]
    assert messages[-2].status == "error"
    assert "业务工具执行失败" in messages[-1].content
    assert "没有生成可用数值结果" in messages[-1].content


def test_system_prompt_defines_tool_selection_and_separates_both_irr_types():
    required_rules = [
        "calculate_economic_metrics",
        "predict_irr",
        "explain_irr_prediction",
        "simulate_scenarios",
        "compare_wind_projects",
        "generate_wind_project_report",
        "公式计算IRR",
        "模型预测IRR",
        "禁止自行心算、估算或编造",
    ]

    for rule in required_rules:
        assert rule in OILWIND_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_graph_recovers_explicit_parameters_omitted_by_model(monkeypatch, tmp_path):
    screenshot_prompt = (
        "这是一个虚构的新项目：装机容量80MW，单位投资5800元/kW，"
        "年利用小时2800小时，上网电价0.42元/kWh，OPEX率0.025。"
    )
    expected_project = {
        "installed_capacity_mw": 80,
        "unit_capex_yuan_per_kw": 5_800,
        "full_load_hours_per_year": 2_800,
        "tariff_yuan_per_kwh": 0.42,
        "opex_ratio": 0.025,
    }
    model = FakeToolModel(
        [
            tool_call(
                "validate_project_parameters",
                {"project": {"project_id": "虚"}},
                "truncated-validation",
            ),
            AIMessage(content="参数校验已通过。"),
        ]
    )
    graph = build_oilwind_agent()

    with patch("oilwind_agent.agent.graph.get_model", return_value=model):
        result = await graph.ainvoke({"messages": [HumanMessage(content=screenshot_prompt)]})

    validation_message = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name == "validate_project_parameters"
    )
    payload = tool_payload(validation_message)
    normalized = payload["normalized_parameters"]
    assert payload["can_calculate"] is True
    for field, value in expected_project.items():
        assert normalized[field] == value
