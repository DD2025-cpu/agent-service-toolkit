"""Structured project-parameter validation for Agent tool calls."""

from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from oilwind_agent.domain import ProjectParameters

REQUIRED_FIELDS = (
    "installed_capacity_mw",
    "full_load_hours_per_year",
    "tariff_yuan_per_kwh",
    "opex_ratio",
)
INVESTMENT_REQUIREMENT = "total_investment_yuan_or_unit_capex_yuan_per_kw"
DEFAULT_ASSUMPTION_FIELDS = (
    "currency",
    "project_life_years",
    "discount_rate",
    "annual_degradation_rate",
    "residual_value_rate",
    "corporate_income_tax_rate",
    "other_revenue_tax_rate",
    "carbon_price_yuan_per_ton",
    "emission_reduction_t_per_mwh",
)

FIELD_LABELS = {
    "installed_capacity_mw": "装机容量",
    "total_investment_yuan": "总投资",
    "unit_capex_yuan_per_kw": "单位投资",
    "full_load_hours_per_year": "年利用小时",
    "tariff_yuan_per_kwh": "上网电价",
    "opex_ratio": "年运维成本率",
    "project_life_years": "项目寿命",
    "discount_rate": "折现率",
    "annual_degradation_rate": "发电量衰减率",
    "residual_value_rate": "残值率",
    "corporate_income_tax_rate": "企业所得税率",
    "other_revenue_tax_rate": "其他收入税率",
    "carbon_price_yuan_per_ton": "碳价",
    "emission_reduction_t_per_mwh": "单位发电减排量",
}
FIELD_REQUIREMENTS = {
    "installed_capacity_mw": "必须大于0且不超过10000 MW",
    "total_investment_yuan": "必须大于0且不超过1万亿元",
    "unit_capex_yuan_per_kw": "必须大于0且不超过100000元/kW",
    "full_load_hours_per_year": "必须大于0且不超过8760小时/年",
    "tariff_yuan_per_kwh": "必须在0至10元/kWh之间",
    "opex_ratio": "必须使用0至1之间的小数比例",
    "project_life_years": "必须是1至50之间的整数年",
    "discount_rate": "必须使用0至1之间的小数比例",
    "annual_degradation_rate": "必须使用0至1之间的小数比例，且不能等于1",
    "residual_value_rate": "必须使用0至1之间的小数比例",
    "corporate_income_tax_rate": "必须使用0至1之间的小数比例",
    "other_revenue_tax_rate": "必须使用0至1之间的小数比例",
    "carbon_price_yuan_per_ton": "必须在0至10000元/吨之间",
    "emission_reduction_t_per_mwh": "必须在0至10吨/MWh之间",
}
LEGACY_OR_WRONG_UNIT_FIELDS = {
    "capacity_mw": "请改用installed_capacity_mw，单位仍为MW。",
    "capacity_kw": "请改用installed_capacity_mw，并先将kW除以1000换算为MW。",
    "investment_million_cny": "请改用total_investment_yuan，并将百万元乘以1000000换算为元。",
    "investment_billion_vnd": "不接受越南盾投资，请提供人民币总投资或人民币单位投资。",
    "full_hours": "请改用full_load_hours_per_year，单位为小时/年。",
    "tariff_cny_per_kwh": "请改用tariff_yuan_per_kwh，单位为元/kWh。",
    "tariff_vnd_per_kwh": "不接受越南盾电价，请提供人民币电价tariff_yuan_per_kwh。",
}
MISSING_QUESTIONS = {
    "installed_capacity_mw": "请提供项目装机容量，单位为MW。",
    "full_load_hours_per_year": "请提供年等效满负荷利用小时，单位为小时/年。",
    "tariff_yuan_per_kwh": "请提供上网电价，单位为元/kWh。",
    "opex_ratio": "请提供年运维成本率，并使用小数表示，例如3%填写0.03。",
    INVESTMENT_REQUIREMENT: ("请提供总投资（元）或单位投资（元/kW）中的至少一项。"),
}


class ParameterIssue(BaseModel):
    """One field-level problem that the Agent can explain to the user."""

    field: str
    message: str
    received: Any | None = None


class ProjectValidationResult(BaseModel):
    """Machine-readable validation and conversation-routing result."""

    valid: bool
    can_calculate: bool
    next_action: Literal["continue_economic_calculation", "ask_user_for_parameters"]
    missing_required: list[str]
    invalid_parameters: list[ParameterIssue]
    unit_errors: list[ParameterIssue]
    defaults_used: dict[str, Any]
    normalized_parameters: dict[str, Any] | None
    follow_up_questions: list[str]


class ProjectParameterValidationError(ValueError):
    """Raised when a calculation is attempted with incomplete project input."""

    def __init__(self, result: ProjectValidationResult):
        self.result = result
        super().__init__("Project parameters are incomplete or invalid.")


class PartialProjectParameters(BaseModel):
    """Agent-facing schema that keeps every project field optional."""

    model_config = ConfigDict(extra="allow")

    installed_capacity_mw: Any | None = Field(default=None, description="装机容量，单位MW。")
    total_investment_yuan: Any | None = Field(default=None, description="总投资，单位人民币元。")
    unit_capex_yuan_per_kw: Any | None = Field(
        default=None,
        description="单位投资，单位人民币元/kW。",
    )
    full_load_hours_per_year: Any | None = Field(
        default=None,
        description="年等效满负荷利用小时，单位小时/年。",
    )
    tariff_yuan_per_kwh: Any | None = Field(
        default=None,
        description="上网电价，单位人民币元/kWh。",
    )
    opex_ratio: Any | None = Field(
        default=None,
        description="年运维成本率，小数比例，例如3%填写0.03。",
    )
    currency: Any | None = Field(default=None, description="货币，只允许CNY。")
    project_id: Any | None = None
    province: Any | None = None
    project_life_years: Any | None = None
    discount_rate: Any | None = None
    annual_degradation_rate: Any | None = None
    residual_value_rate: Any | None = None
    corporate_income_tax_rate: Any | None = None
    other_revenue_tax_rate: Any | None = None
    carbon_price_yuan_per_ton: Any | None = None
    emission_reduction_t_per_mwh: Any | None = None


class ValidateProjectParametersInput(BaseModel):
    """Partial input is intentional so missing fields can be reported together."""

    project: PartialProjectParameters = Field(
        description=(
            "已知的风电项目参数。货币必须使用人民币；允许参数不完整，"
            "工具会返回需要向用户追问的字段。"
        )
    )


def _default_assumptions(data: Mapping[str, Any]) -> dict[str, Any]:
    defaults = {}
    for field_name in DEFAULT_ASSUMPTION_FIELDS:
        if field_name not in data:
            defaults[field_name] = ProjectParameters.model_fields[field_name].default
    return defaults


def _missing_required(data: Mapping[str, Any]) -> list[str]:
    missing = [name for name in REQUIRED_FIELDS if data.get(name) is None]
    if data.get("total_investment_yuan") is None and data.get("unit_capex_yuan_per_kw") is None:
        missing.append(INVESTMENT_REQUIREMENT)
    return missing


def _unit_errors(data: Mapping[str, Any]) -> list[ParameterIssue]:
    issues = [
        ParameterIssue(field=field, message=message, received=data[field])
        for field, message in LEGACY_OR_WRONG_UNIT_FIELDS.items()
        if field in data
    ]

    currency = data.get("currency")
    if currency is not None and currency != "CNY":
        issues.append(
            ParameterIssue(
                field="currency",
                message="所有外部货币输入必须使用人民币，currency必须为CNY。",
                received=currency,
            )
        )

    opex = data.get("opex_ratio")
    if isinstance(opex, (int, float)) and not isinstance(opex, bool) and 1 < opex <= 100:
        issues.append(
            ParameterIssue(
                field="opex_ratio",
                message="opex_ratio必须使用小数比例，例如3%填写0.03，不能填写3。",
                received=opex,
            )
        )

    tariff = data.get("tariff_yuan_per_kwh")
    if isinstance(tariff, (int, float)) and not isinstance(tariff, bool) and tariff > 10:
        issues.append(
            ParameterIssue(
                field="tariff_yuan_per_kwh",
                message="电价疑似使用了越南盾或分，请换算为人民币元/kWh。",
                received=tariff,
            )
        )
    return issues


def _validation_issues(
    data: Mapping[str, Any],
    unit_issue_fields: set[str],
) -> tuple[list[ParameterIssue], ProjectParameters | None]:
    try:
        return [], ProjectParameters.model_validate(dict(data))
    except ValidationError as error:
        issues = []
        for detail in error.errors(include_url=False):
            location = detail.get("loc", ())
            field = str(location[0]) if location else "investment"
            if detail["type"] == "missing" or field in unit_issue_fields:
                continue
            requirement = FIELD_REQUIREMENTS.get(field)
            message = requirement or detail["msg"].removeprefix("Value error, ")
            issues.append(
                ParameterIssue(
                    field=field,
                    message=f"{FIELD_LABELS.get(field, field)}：{message}",
                    received=detail.get("input") if field != "investment" else None,
                )
            )
        return issues, None


def _follow_up_questions(
    missing: list[str],
    invalid: list[ParameterIssue],
    unit_errors: list[ParameterIssue],
) -> list[str]:
    questions = [MISSING_QUESTIONS[field] for field in missing]
    questions.extend(f"请修正{issue.message}" for issue in unit_errors)
    questions.extend(f"请重新提供{issue.message}" for issue in invalid)
    return list(dict.fromkeys(questions))


def validate_project_parameters(project: Mapping[str, Any]) -> ProjectValidationResult:
    """Validate partial CNY input without inventing missing project facts."""
    if not isinstance(project, Mapping):
        issue = ParameterIssue(
            field="project",
            message="项目参数必须是字段映射对象。",
            received=project,
        )
        return ProjectValidationResult(
            valid=False,
            can_calculate=False,
            next_action="ask_user_for_parameters",
            missing_required=[],
            invalid_parameters=[issue],
            unit_errors=[],
            defaults_used={},
            normalized_parameters=None,
            follow_up_questions=["请按字段名称提供项目参数。"],
        )

    missing = _missing_required(project)
    unit_errors = _unit_errors(project)
    invalid, normalized = _validation_issues(
        project,
        {issue.field for issue in unit_errors},
    )
    valid = not missing and not invalid and not unit_errors and normalized is not None
    normalized_parameters = normalized.model_dump(mode="json") if valid else None

    return ProjectValidationResult(
        valid=valid,
        can_calculate=valid,
        next_action=("continue_economic_calculation" if valid else "ask_user_for_parameters"),
        missing_required=missing,
        invalid_parameters=invalid,
        unit_errors=unit_errors,
        defaults_used=_default_assumptions(project),
        normalized_parameters=normalized_parameters,
        follow_up_questions=_follow_up_questions(missing, invalid, unit_errors),
    )


def require_valid_project_parameters(
    project: Mapping[str, Any],
) -> ProjectParameters:
    """Guard used by every economic tool before it performs a calculation."""
    result = validate_project_parameters(project)
    if not result.can_calculate or result.normalized_parameters is None:
        raise ProjectParameterValidationError(result)
    return ProjectParameters.model_validate(result.normalized_parameters)


@tool("validate_project_parameters", args_schema=ValidateProjectParametersInput)
def _validate_project_parameters_tool(
    project: PartialProjectParameters,
) -> dict[str, Any]:
    """检查风电项目人民币参数；若不完整，返回缺失项和追问问题。

    必须在经济计算、模型预测和情景模拟前调用。仅当返回的
    can_calculate为true时才能继续；否则应按follow_up_questions向用户追问。
    """
    project_data = project.model_dump(exclude_none=True)
    return validate_project_parameters(project_data).model_dump(mode="json")


validate_project_parameters_tool: BaseTool = _validate_project_parameters_tool
