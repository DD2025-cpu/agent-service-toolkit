"""LangGraph-ready wind-project business tools with structured JSON results."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from oilwind_agent.domain.economics import (
    calculate_economic_metrics,
    economic_assumptions,
)
from oilwind_agent.domain.scenarios import SCENARIO_DEFINITIONS, simulate_scenarios
from oilwind_agent.model.artifacts import (
    DEFAULT_MODEL_NAME,
    ModelArtifactError,
    explain_irr,
    predict_irr,
)
from oilwind_agent.model.feature_adapter import adapt_project_to_irr_features
from oilwind_agent.tools.parameter_validation import (
    PartialProjectParameters,
    ProjectParameterValidationError,
    require_valid_project_parameters,
)

TOOL_RESULT_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MODEL_SCOPE_WARNING = (
    "本模型使用越南公开风电项目样本经固定汇率换算、人民币字段统一和既定经济口径"
    "处理后形成的人民币口径代理数据集训练；结果用于论文的方法验证、基线比较和"
    "预评价，不代表基于真实中国风电项目样本训练。项目评价应以现金流公式测算和"
    "实际项目尽调数据为准。"
)


class ProjectToolInput(BaseModel):
    project: PartialProjectParameters


class ModelProjectToolInput(ProjectToolInput):
    model_name: str = Field(default=DEFAULT_MODEL_NAME)


class ScenarioToolInput(ProjectToolInput):
    scenario_names: list[str] | None = Field(
        default=None,
        description=(
            "合法名称：pessimistic、baseline、optimistic、tariff_low、tariff_high、"
            "full_hours_low、full_hours_high、opex_low、opex_high、carbon_50、"
            "carbon_100、combined_stress、combined_upside。基准情景必须使用baseline。"
        ),
    )
    random_state: int = Field(default=42)


class CompareProjectsInput(BaseModel):
    projects: list[PartialProjectParameters] = Field(min_length=2)
    model_name: str = Field(default=DEFAULT_MODEL_NAME)


class ReportToolInput(ModelProjectToolInput):
    include_scenarios: bool = True
    scenario_names: list[str] | None = Field(
        default=None,
        description="使用simulate_scenarios支持的英文情景名称；省略表示包含全部情景。",
    )


def _project_data(project: PartialProjectParameters | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(project, PartialProjectParameters):
        return project.model_dump(exclude_none=True)
    return dict(project)


def _model_info(metadata: Mapping[str, Any]) -> dict[str, Any]:
    training = metadata.get("training", {})
    return {
        "name": metadata["model_name"],
        "version": metadata["model_version"],
        "target": metadata["target_column"],
        "feature_columns": metadata["feature_columns"],
        "artifact_sha256": metadata["model_sha256"],
        "training_data_file": training.get("training_data_file"),
        "test_metrics": training.get("metrics", {}).get("test"),
    }


def _success(tool_name: str, **content: Any) -> dict[str, Any]:
    return {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "tool": tool_name,
        "status": "success",
        **content,
    }


def _needs_input(tool_name: str, error: ProjectParameterValidationError) -> dict[str, Any]:
    return {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "tool": tool_name,
        "status": "needs_input",
        "validation": error.result.model_dump(mode="json"),
    }


def _tool_error(tool_name: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "tool": tool_name,
        "status": "error",
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def calculate_economic_metrics_result(project: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and calculate deterministic economic indicators."""
    normalized = require_valid_project_parameters(project)
    return _success(
        "calculate_economic_metrics",
        project=normalized.model_dump(mode="json"),
        assumptions=economic_assumptions(normalized),
        metrics=calculate_economic_metrics(normalized),
        warnings=[],
    )


def predict_irr_result(
    project: Mapping[str, Any],
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Load a packaged XGBoost model and predict IRR without retraining."""
    normalized = require_valid_project_parameters(project)
    features = adapt_project_to_irr_features(normalized)
    prediction, metadata = predict_irr(features, model_name=model_name)
    warnings = [MODEL_SCOPE_WARNING]
    if normalized.carbon_price_yuan_per_ton:
        warnings.append("模型不含碳价特征，预测IRR未计入碳收益。")
    return _success(
        "predict_irr",
        project_id=normalized.project_id,
        predicted_irr=prediction,
        predicted_irr_percent=prediction * 100,
        model=_model_info(metadata),
        model_features=features.as_ordered_dict(),
        warnings=warnings,
    )


def explain_irr_prediction_result(
    project: Mapping[str, Any],
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Return exact local TreeSHAP contributions for one model prediction."""
    normalized = require_valid_project_parameters(project)
    features = adapt_project_to_irr_features(normalized)
    explanation, metadata = explain_irr(features, model_name=model_name)
    return _success(
        "explain_irr_prediction",
        project_id=normalized.project_id,
        model=_model_info(metadata),
        model_features=features.as_ordered_dict(),
        explanation=explanation,
        warnings=[MODEL_SCOPE_WARNING],
    )


def simulate_scenarios_result(
    project: Mapping[str, Any],
    scenario_names: Sequence[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run reproducible named economic scenarios."""
    normalized = require_valid_project_parameters(project)
    results = simulate_scenarios(normalized, scenario_names, random_state)
    valid_irrs = [item for item in results if item["metrics"]["irr"] is not None]
    summary = {
        "best_scenario_by_irr": (
            max(valid_irrs, key=lambda item: item["metrics"]["irr"])["scenario"]
            if valid_irrs
            else None
        ),
        "worst_scenario_by_irr": (
            min(valid_irrs, key=lambda item: item["metrics"]["irr"])["scenario"]
            if valid_irrs
            else None
        ),
    }
    return _success(
        "simulate_scenarios",
        project_id=normalized.project_id,
        random_state=int(random_state),
        sampling_method="same seed and sampling order for every scenario",
        scenario_count=len(results),
        available_scenarios=list(SCENARIO_DEFINITIONS),
        results=results,
        summary=summary,
        warnings=["碳收益仅为研究情景，不代表项目已经取得碳信用签发资格。"],
    )


def compare_projects_result(
    projects: Sequence[Mapping[str, Any]],
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Compare two or more projects under one calculation and model contract."""
    if len(projects) < 2:
        raise ValueError("项目对比至少需要两个项目。")
    rows = []
    for index, project in enumerate(projects, start=1):
        normalized = require_valid_project_parameters(project)
        metrics = calculate_economic_metrics(normalized)
        features = adapt_project_to_irr_features(normalized)
        prediction, metadata = predict_irr(features, model_name=model_name)
        rows.append(
            {
                "project_id": normalized.project_id or f"project_{index}",
                "calculated_irr": metrics["irr"],
                "predicted_irr": prediction,
                "npv_yuan": metrics["npv_yuan"],
                "static_payback_years": metrics["static_payback_years"],
                "discounted_payback_years": metrics["discounted_payback_years"],
                "lcoe_yuan_per_kwh": metrics["lcoe_yuan_per_kwh"],
            }
        )
    ranked = sorted(
        rows,
        key=lambda item: (
            item["calculated_irr"] is not None,
            item["calculated_irr"] if item["calculated_irr"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank_by_calculated_irr"] = rank
    return _success(
        "compare_projects",
        ranking_basis="calculated_irr",
        project_count=len(ranked),
        projects=ranked,
        model=_model_info(metadata),
        warnings=[MODEL_SCOPE_WARNING],
    )


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_.")
    return cleaned or "wind_project"


def _percent(value: float | None) -> str:
    return "无法计算" if value is None else f"{value:.4%}"


def _number(value: float | None, decimals: int = 2) -> str:
    return "无法计算" if value is None else f"{value:,.{decimals}f}"


def generate_project_report_result(
    project: Mapping[str, Any],
    model_name: str = DEFAULT_MODEL_NAME,
    include_scenarios: bool = True,
    scenario_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate a Markdown report under the project's controlled output directory."""
    normalized = require_valid_project_parameters(project)
    economic = calculate_economic_metrics_result(project)
    prediction = predict_irr_result(project, model_name)
    explanation = explain_irr_prediction_result(project, model_name)
    scenarios = simulate_scenarios_result(project, scenario_names) if include_scenarios else None
    metrics = economic["metrics"]
    project_name = normalized.project_id or "未命名项目"
    lines = [
        f"# {project_name} 风电项目经济评价报告",
        "",
        "## 项目输入",
        "",
        f"- 装机容量：{normalized.installed_capacity_mw:.3f} MW",
        f"- 总投资：{normalized.total_investment_yuan:,.2f} 元",
        f"- 单位投资：{normalized.unit_capex_yuan_per_kw:,.2f} 元/kW",
        f"- 年利用小时：{normalized.full_load_hours_per_year:.2f} 小时/年",
        f"- 上网电价：{normalized.tariff_yuan_per_kwh:.4f} 元/kWh",
        f"- 年运维成本率：{normalized.opex_ratio:.4%}",
        "",
        "## 经济指标",
        "",
        f"- 计算IRR：{_percent(metrics['irr'])}",
        f"- NPV：{_number(metrics['npv_yuan'])} 元",
        f"- 静态回收期：{_number(metrics['static_payback_years'], 3)} 年",
        f"- 动态回收期：{_number(metrics['discounted_payback_years'], 3)} 年",
        f"- LCOE：{_number(metrics['lcoe_yuan_per_kwh'], 4)} 元/kWh",
        "",
        "## 模型预测与解释",
        "",
        f"- 预测IRR：{_percent(prediction['predicted_irr'])}",
        f"- 模型：{prediction['model']['name']} {prediction['model']['version']}",
        "",
        "| 特征 | 特征值 | SHAP贡献 | 方向 |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in explanation["explanation"]["contributions"]:
        lines.append(
            f"| {item['feature']} | {item['feature_value']:.6g} | "
            f"{item['shap_value']:.6f} | {item['effect']} |"
        )
    if scenarios is not None:
        lines.extend(
            [
                "",
                "## 情景模拟",
                "",
                "| 情景 | IRR | NPV（元） | LCOE（元/kWh） |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for item in scenarios["results"]:
            scenario_metrics = item["metrics"]
            lines.append(
                f"| {item['label']} | {_percent(scenario_metrics['irr'])} | "
                f"{_number(scenario_metrics['npv_yuan'])} | "
                f"{_number(scenario_metrics['lcoe_yuan_per_kwh'], 4)} |"
            )
    lines.extend(
        [
            "",
            "## 口径和适用范围",
            "",
            f"- {MODEL_SCOPE_WARNING}",
            "- SHAP贡献反映模型关系，不代表因果关系。",
            "- 碳收益属于研究情景，不代表碳信用签发资格。",
            "",
        ]
    )
    report_markdown = "\n".join(lines)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{_slug(project_name)}_report.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    return _success(
        "generate_project_report",
        project_id=normalized.project_id,
        report_path=str(report_path),
        report_markdown=report_markdown,
        included_scenarios=include_scenarios,
        warnings=[MODEL_SCOPE_WARNING],
    )


def _run_tool(tool_name: str, function: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return function(*args, **kwargs)
    except ProjectParameterValidationError as error:
        return _needs_input(tool_name, error)
    except (ModelArtifactError, OSError, ValueError) as error:
        return _tool_error(tool_name, error)
    except Exception as error:
        return _tool_error(tool_name, error)


@tool("calculate_economic_metrics", args_schema=ProjectToolInput)
def calculate_economic_metrics_tool(project: PartialProjectParameters) -> dict[str, Any]:
    """计算风电项目IRR、NPV、静态/动态回收期和LCOE；参数不足时返回追问项。"""
    return _run_tool(
        "calculate_economic_metrics",
        calculate_economic_metrics_result,
        _project_data(project),
    )


@tool("predict_irr", args_schema=ModelProjectToolInput)
def predict_irr_tool(
    project: PartialProjectParameters,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """加载已保存的XGBoost模型预测IRR，不重新训练；同时返回模型版本和特征。"""
    return _run_tool("predict_irr", predict_irr_result, _project_data(project), model_name)


@tool("explain_irr_prediction", args_schema=ModelProjectToolInput)
def explain_irr_prediction_tool(
    project: PartialProjectParameters,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """使用与预测模型相同的工件返回单项目精确TreeSHAP特征贡献。"""
    return _run_tool(
        "explain_irr_prediction",
        explain_irr_prediction_result,
        _project_data(project),
        model_name,
    )


@tool("simulate_scenarios", args_schema=ScenarioToolInput)
def simulate_scenarios_tool(
    project: PartialProjectParameters,
    scenario_names: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """运行悲观、基准、乐观、电价、利用小时、OPEX、碳价和组合情景。"""
    return _run_tool(
        "simulate_scenarios",
        simulate_scenarios_result,
        _project_data(project),
        scenario_names,
        random_state,
    )


@tool("compare_wind_projects", args_schema=CompareProjectsInput)
def compare_wind_projects_tool(
    projects: list[PartialProjectParameters],
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """按统一人民币口径计算并比较两个及以上风电项目。"""
    project_data = [_project_data(project) for project in projects]
    return _run_tool("compare_wind_projects", compare_projects_result, project_data, model_name)


@tool("generate_wind_project_report", args_schema=ReportToolInput)
def generate_wind_project_report_tool(
    project: PartialProjectParameters,
    model_name: str = DEFAULT_MODEL_NAME,
    include_scenarios: bool = True,
    scenario_names: list[str] | None = None,
) -> dict[str, Any]:
    """生成包含经济指标、IRR预测、SHAP解释和可选情景结果的Markdown报告。"""
    return _run_tool(
        "generate_wind_project_report",
        generate_project_report_result,
        _project_data(project),
        model_name,
        include_scenarios,
        scenario_names,
    )


BUSINESS_TOOLS: list[BaseTool] = [
    calculate_economic_metrics_tool,
    predict_irr_tool,
    explain_irr_prediction_tool,
    simulate_scenarios_tool,
    compare_wind_projects_tool,
    generate_wind_project_report_tool,
]
