"""Deterministic recovery of explicit project values from Chinese user text."""

import re
from typing import Any

NUMBER = r"([0-9]+(?:\.[0-9]+)?)"


def _number(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def extract_explicit_project_parameters(text: str) -> dict[str, Any]:
    """Extract only values that are explicitly present with recognizable units."""
    result: dict[str, Any] = {}

    project_id = re.search(
        r"虚构的([\u4e00-\u9fffA-Za-z0-9_-]+?项目[A-Za-z0-9_-]+)(?:做|进行|，|,|：|:)",
        text,
    )
    if project_id:
        result["project_id"] = project_id.group(1)

    capacity = re.search(
        rf"装机容量\s*[:：]?\s*{NUMBER}\s*(MW|兆瓦|kW|千瓦)",
        text,
        flags=re.IGNORECASE,
    )
    if capacity:
        value = float(capacity.group(1))
        unit = capacity.group(2).lower()
        if unit in {"mw", "兆瓦"}:
            result["installed_capacity_mw"] = value
        else:
            result["capacity_kw"] = value

    unit_capex = _number(
        text,
        rf"单位投资\s*[:：]?\s*{NUMBER}\s*元\s*/\s*kW",
    )
    if unit_capex is not None:
        result["unit_capex_yuan_per_kw"] = unit_capex

    total_investment = re.search(
        rf"总投资\s*[:：]?\s*{NUMBER}\s*(亿元|万元|元)",
        text,
        flags=re.IGNORECASE,
    )
    if total_investment:
        multiplier = {"亿元": 100_000_000, "万元": 10_000, "元": 1}[total_investment.group(2)]
        result["total_investment_yuan"] = float(total_investment.group(1)) * multiplier

    full_hours = _number(
        text,
        rf"(?:年利用小时|年等效满负荷利用小时)\s*[:：]?\s*{NUMBER}\s*(?:小时|h)",
    )
    if full_hours is not None:
        result["full_load_hours_per_year"] = full_hours

    tariff = _number(
        text,
        rf"(?:上网电价|电价)\s*[:：]?\s*{NUMBER}\s*元\s*/\s*kWh",
    )
    if tariff is not None:
        result["tariff_yuan_per_kwh"] = tariff

    opex = re.search(
        rf"(?:OPEX率|OPEX|年运维成本率)\s*(?:填|为|=|:|：)?\s*{NUMBER}\s*(%)?",
        text,
        flags=re.IGNORECASE,
    )
    if opex:
        value = float(opex.group(1))
        result["opex_ratio"] = value / 100 if opex.group(2) else value

    return result


def extract_from_human_messages(messages: list[Any]) -> dict[str, Any]:
    """Merge explicit values across turns, with later user values taking precedence."""
    extracted: dict[str, Any] = {}
    for message in messages:
        if getattr(message, "type", None) == "human" and isinstance(message.content, str):
            extracted.update(extract_explicit_project_parameters(message.content))
    return extracted


def enrich_validation_tool_calls(response: Any, messages: list[Any]) -> Any:
    """Fill model-omitted validation fields from explicit values in the user message."""
    extracted = extract_from_human_messages(messages)
    if not extracted or not response.tool_calls:
        return response

    validation_call_count = sum(
        tool_call.get("name") == "validate_project_parameters" for tool_call in response.tool_calls
    )
    updated_calls = []
    changed = False
    for tool_call in response.tool_calls:
        if tool_call.get("name") != "validate_project_parameters":
            updated_calls.append(tool_call)
            continue
        args = dict(tool_call.get("args", {}))
        model_project = args.get("project")
        project = dict(model_project) if isinstance(model_project, dict) else {}
        has_required_fields = all(
            project.get(field) is not None
            for field in (
                "installed_capacity_mw",
                "full_load_hours_per_year",
                "tariff_yuan_per_kwh",
                "opex_ratio",
            )
        ) and (
            project.get("total_investment_yuan") is not None
            or project.get("unit_capex_yuan_per_kw") is not None
        )

        if validation_call_count > 1:
            updated_calls.append(tool_call)
            continue
        if has_required_fields:
            explicit_project_id = extracted.get("project_id")
            if explicit_project_id and explicit_project_id != project.get("project_id"):
                project["project_id"] = explicit_project_id
                args["project"] = project
                updated_calls.append({**tool_call, "args": args})
                changed = True
            else:
                updated_calls.append(tool_call)
            continue
        if project.get("project_id") in {"虚", "新"} and "虚构的新项目" in " ".join(
            message.content
            for message in messages
            if getattr(message, "type", None) == "human" and isinstance(message.content, str)
        ):
            project.pop("project_id")
        if "capacity_kw" in extracted:
            project.pop("installed_capacity_mw", None)
        project.update(extracted)
        args["project"] = project
        updated_calls.append({**tool_call, "args": args})
        changed = True
    return response.model_copy(update={"tool_calls": updated_calls}) if changed else response
