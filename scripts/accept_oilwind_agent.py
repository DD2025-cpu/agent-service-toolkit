"""Run live DeepSeek acceptance cases for the dedicated OilWind Agent."""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from oilwind_agent.agent import build_oilwind_agent  # noqa: E402

PROJECT_TEXT = (
    "项目A：装机容量50MW，单位投资5200元/kW，年利用小时2400小时，上网电价0.35元/kWh，OPEX率0.03。"
)


@dataclass(frozen=True)
class AcceptanceCase:
    name: str
    prompt: str
    expected_tool: str


CASES = [
    AcceptanceCase(
        "公式IRR", f"{PROJECT_TEXT} 请只做现金流公式IRR计算。", "calculate_economic_metrics"
    ),
    AcceptanceCase("模型预测", f"{PROJECT_TEXT} 请用已保存的XGBoost模型预测IRR。", "predict_irr"),
    AcceptanceCase(
        "SHAP解释", f"{PROJECT_TEXT} 请解释模型预测，给出SHAP贡献。", "explain_irr_prediction"
    ),
    AcceptanceCase(
        "情景分析",
        f"{PROJECT_TEXT} 请分析悲观、基准和乐观三种情景。",
        "simulate_scenarios",
    ),
    AcceptanceCase(
        "项目比较",
        (f"{PROJECT_TEXT} 项目B除电价为0.45元/kWh外，其余参数与A相同。请比较两个项目。"),
        "compare_wind_projects",
    ),
    AcceptanceCase(
        "非法单位与异常值",
        (
            "装机容量50000kW，单位投资5200元/kW，年利用小时2400小时，"
            "电价1500元/kWh，OPEX率填3，请计算IRR。"
        ),
        "validate_project_parameters",
    ),
    AcceptanceCase(
        "完整项目评价",
        f"{PROJECT_TEXT} 请完成综合评价并保存报告。",
        "generate_wind_project_report",
    ),
]


def _tool_names(messages) -> list[str]:
    return [message.name for message in messages if isinstance(message, ToolMessage)]


def _last_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content).replace("\n", " ")[:240]
    return ""


def _tool_payload(message: ToolMessage) -> dict:
    if isinstance(message.artifact, dict):
        return message.artifact
    if isinstance(message.content, str):
        return json.loads(message.content)
    raise TypeError(f"Unexpected tool payload type: {type(message.content)}")


async def _run_case(graph, case: AcceptanceCase, index: int) -> None:
    config = {"configurable": {"thread_id": f"live-acceptance-{index}"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=case.prompt)]},
        config,
    )
    names = _tool_names(result["messages"])
    if case.expected_tool not in names:
        raise AssertionError(f"{case.name}: expected {case.expected_tool}, got {names}")
    expected_message = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name == case.expected_tool
    )
    payload = _tool_payload(expected_message)
    if case.name == "非法单位与异常值":
        if payload.get("can_calculate") is not False:
            raise AssertionError(f"{case.name}: invalid input was not blocked: {payload}")
    elif payload.get("status") != "success":
        raise AssertionError(f"{case.name}: tool did not succeed: {payload}")
    if case.name == "完整项目评价" and payload.get("project_id") != "项目A":
        raise AssertionError(
            f"{case.name}: project_id was not preserved: {payload.get('project_id')}"
        )
    print(f"PASS {case.name}: {' -> '.join(names)}")
    print(f"  {_last_answer(result['messages'])}")


async def _run_missing_then_continue(graph) -> None:
    config = {"configurable": {"thread_id": "live-acceptance-follow-up"}}
    first = await graph.ainvoke(
        {"messages": [HumanMessage(content="项目装机容量50MW，请计算公式IRR。")]},
        config,
    )
    first_names = _tool_names(first["messages"])
    first_answer = _last_answer(first["messages"])
    if first_names != ["validate_project_parameters"] or "暂停计算" not in first_answer:
        raise AssertionError(f"缺参追问失败: tools={first_names}, answer={first_answer}")

    second = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "补充：单位投资5200元/kW，年利用小时2400小时，"
                        "上网电价0.35元/kWh，OPEX率0.03。"
                    )
                )
            ]
        },
        config,
    )
    second_names = _tool_names(second["messages"])
    if "calculate_economic_metrics" not in second_names:
        raise AssertionError(f"补参后未继续计算: tools={second_names}")
    print(f"PASS 缺参后继续: {' -> '.join(second_names)}")
    print(f"  {_last_answer(second['messages'])}")


async def main() -> None:
    graph = build_oilwind_agent(MemorySaver())
    selected = list(enumerate(CASES, start=1))
    if len(sys.argv) == 2:
        case_number = int(sys.argv[1])
        selected = [(case_number, CASES[case_number - 1])]
    for index, case in selected:
        await _run_case(graph, case, index)
    if len(sys.argv) == 1:
        await _run_missing_then_continue(graph)


if __name__ == "__main__":
    asyncio.run(main())
