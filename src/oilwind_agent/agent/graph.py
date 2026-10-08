"""Dedicated LangGraph workflow for wind-project economic evaluation."""

import ast
import json
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from core import get_model, settings
from oilwind_agent.agent.parameter_extraction import enrich_validation_tool_calls
from oilwind_agent.agent.prompts import OILWIND_SYSTEM_PROMPT
from oilwind_agent.tools import ALL_OILWIND_TOOLS


class OilWindAgentState(MessagesState, total=False):
    remaining_steps: RemainingSteps


def _wrap_model(model: BaseChatModel) -> RunnableSerializable[OilWindAgentState, AIMessage]:
    bound_model = model.bind_tools(ALL_OILWIND_TOOLS)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=OILWIND_SYSTEM_PROMPT)] + state["messages"],
        name="OilWindSystemPrompt",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


async def call_model(
    state: OilWindAgentState,
    config: RunnableConfig,
) -> OilWindAgentState:
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await _wrap_model(model).ainvoke(state, config)
    response = enrich_validation_tool_calls(response, state["messages"])
    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content=(
                        "工作流剩余步骤不足，尚未执行工具，也没有生成任何数值结果。请重新提交请求。"
                    ),
                )
            ]
        }
    return {"messages": [response]}


def route_after_model(state: OilWindAgentState) -> Literal["tools", "done"]:
    message = state["messages"][-1]
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(message)}")
    return "tools" if message.tool_calls else "done"


def _tool_payload(message: ToolMessage) -> dict[str, Any] | None:
    content = message.content
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return None
    return value if isinstance(value, dict) else None


def _trailing_tool_payloads(state: OilWindAgentState) -> list[dict[str, Any]]:
    payloads = []
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            break
        payload = _tool_payload(message)
        if payload is not None:
            payloads.append(payload)
    return list(reversed(payloads))


def route_after_tools(
    state: OilWindAgentState,
) -> Literal["model", "needs_input", "tool_error"]:
    trailing_messages = []
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            break
        trailing_messages.append(message)
    if any(message.status == "error" for message in trailing_messages):
        return "tool_error"

    payloads = _trailing_tool_payloads(state)
    if any(payload.get("status") == "error" for payload in payloads):
        return "tool_error"
    if any(
        payload.get("status") == "needs_input" or payload.get("can_calculate") is False
        for payload in payloads
    ):
        return "needs_input"
    return "model"


def respond_to_missing_parameters(state: OilWindAgentState) -> OilWindAgentState:
    questions: list[str] = []
    for payload in _trailing_tool_payloads(state):
        validation = payload.get("validation", payload)
        questions.extend(validation.get("follow_up_questions", []))
    questions = list(dict.fromkeys(questions))
    if questions:
        content = "当前参数不足，已暂停计算。请补充或修正以下信息：\n\n" + "\n".join(
            f"- {question}" for question in questions
        )
    else:
        content = "当前参数不足，已暂停计算。请补充完整的人民币项目参数。"
    return {"messages": [AIMessage(content=content)]}


def respond_to_tool_error(state: OilWindAgentState) -> OilWindAgentState:
    errors = []
    for payload in _trailing_tool_payloads(state):
        if payload.get("status") != "error":
            continue
        error = payload.get("error", {})
        errors.append(str(error.get("message", "未知工具错误")))
    if not errors:
        for message in reversed(state["messages"]):
            if not isinstance(message, ToolMessage):
                break
            if message.status == "error":
                errors.append(str(message.content))
    details = "；".join(dict.fromkeys(errors)) or "未知工具错误"
    return {
        "messages": [
            AIMessage(
                content=(
                    f"业务工具执行失败：{details}。本次没有生成可用数值结果，请修正输入或稍后重试。"
                )
            )
        ]
    }


def build_oilwind_agent(checkpointer: Any | None = None):
    """Build the dedicated graph; injectable checkpointer supports multi-turn tests."""
    graph = StateGraph(OilWindAgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(ALL_OILWIND_TOOLS))
    graph.add_node("needs_input", respond_to_missing_parameters)
    graph.add_node("tool_error", respond_to_tool_error)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", route_after_model, {"tools": "tools", "done": END})
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"model": "model", "needs_input": "needs_input", "tool_error": "tool_error"},
    )
    graph.add_edge("needs_input", END)
    graph.add_edge("tool_error", END)
    return graph.compile(checkpointer=checkpointer)


oilwind_agent = build_oilwind_agent()
