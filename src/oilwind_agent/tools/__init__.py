"""LangChain tools exposed by OilWind-Agent."""

from oilwind_agent.tools.business import (
    BUSINESS_TOOLS,
    calculate_economic_metrics_result,
    calculate_economic_metrics_tool,
    compare_projects_result,
    compare_wind_projects_tool,
    explain_irr_prediction_result,
    explain_irr_prediction_tool,
    generate_project_report_result,
    generate_wind_project_report_tool,
    predict_irr_result,
    predict_irr_tool,
    simulate_scenarios_result,
    simulate_scenarios_tool,
)
from oilwind_agent.tools.parameter_validation import (
    ProjectParameterValidationError,
    ProjectValidationResult,
    require_valid_project_parameters,
    validate_project_parameters,
    validate_project_parameters_tool,
)

ALL_OILWIND_TOOLS = [validate_project_parameters_tool, *BUSINESS_TOOLS]

__all__ = [
    "ALL_OILWIND_TOOLS",
    "BUSINESS_TOOLS",
    "ProjectParameterValidationError",
    "ProjectValidationResult",
    "calculate_economic_metrics_result",
    "calculate_economic_metrics_tool",
    "compare_projects_result",
    "compare_wind_projects_tool",
    "explain_irr_prediction_result",
    "explain_irr_prediction_tool",
    "generate_project_report_result",
    "generate_wind_project_report_tool",
    "predict_irr_result",
    "predict_irr_tool",
    "require_valid_project_parameters",
    "simulate_scenarios_result",
    "simulate_scenarios_tool",
    "validate_project_parameters",
    "validate_project_parameters_tool",
]
