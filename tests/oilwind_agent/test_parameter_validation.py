import math

import pytest

from oilwind_agent.tools import (
    ProjectParameterValidationError,
    require_valid_project_parameters,
    validate_project_parameters,
    validate_project_parameters_tool,
)

COMPLETE_PROJECT = {
    "installed_capacity_mw": 50,
    "unit_capex_yuan_per_kw": 5_200,
    "full_load_hours_per_year": 2_400,
    "tariff_yuan_per_kwh": 0.35,
    "opex_ratio": 0.03,
}


def test_complete_project_can_continue_and_records_defaults():
    result = validate_project_parameters(COMPLETE_PROJECT)

    assert result.valid is True
    assert result.can_calculate is True
    assert result.next_action == "continue_economic_calculation"
    assert result.missing_required == []
    assert result.unit_errors == []
    assert result.invalid_parameters == []
    assert result.defaults_used == {
        "currency": "CNY",
        "project_life_years": 20,
        "discount_rate": 0.08,
        "annual_degradation_rate": 0.005,
        "residual_value_rate": 0.05,
        "corporate_income_tax_rate": 0.2,
        "other_revenue_tax_rate": 0.0,
        "carbon_price_yuan_per_ton": 0.0,
        "emission_reduction_t_per_mwh": 0.5,
    }
    assert result.normalized_parameters["total_investment_yuan"] == 260_000_000


def test_missing_fields_are_listed_and_generate_follow_up_questions():
    result = validate_project_parameters({"installed_capacity_mw": 50})

    assert result.can_calculate is False
    assert result.next_action == "ask_user_for_parameters"
    assert result.missing_required == [
        "full_load_hours_per_year",
        "tariff_yuan_per_kwh",
        "opex_ratio",
        "total_investment_yuan_or_unit_capex_yuan_per_kw",
    ]
    assert len(result.follow_up_questions) == 4
    assert result.normalized_parameters is None


def test_wrong_units_and_percent_notation_are_identified():
    result = validate_project_parameters(
        {
            "capacity_kw": 50_000,
            "investment_billion_vnd": 873,
            "full_load_hours_per_year": 2_400,
            "tariff_yuan_per_kwh": 1_500,
            "opex_ratio": 3,
        }
    )

    fields = {issue.field for issue in result.unit_errors}
    assert fields == {
        "capacity_kw",
        "investment_billion_vnd",
        "tariff_yuan_per_kwh",
        "opex_ratio",
    }
    assert "installed_capacity_mw" in result.missing_required
    assert result.can_calculate is False


def test_out_of_range_and_non_finite_values_are_rejected():
    result = validate_project_parameters(
        COMPLETE_PROJECT
        | {
            "full_load_hours_per_year": 8_761,
            "unit_capex_yuan_per_kw": math.nan,
        }
    )

    fields = {issue.field for issue in result.invalid_parameters}
    assert fields == {"full_load_hours_per_year", "unit_capex_yuan_per_kw"}
    assert result.can_calculate is False


def test_explicit_assumptions_are_not_reported_as_defaults():
    result = validate_project_parameters(COMPLETE_PROJECT | {"discount_rate": 0.1})

    assert result.can_calculate is True
    assert "discount_rate" not in result.defaults_used
    assert result.normalized_parameters["discount_rate"] == 0.1


def test_calculation_guard_blocks_incomplete_parameters():
    with pytest.raises(ProjectParameterValidationError) as raised:
        require_valid_project_parameters({"installed_capacity_mw": 50})

    assert raised.value.result.can_calculate is False
    assert raised.value.result.follow_up_questions


def test_langchain_tool_accepts_partial_input_for_agent_follow_up():
    result = validate_project_parameters_tool.invoke({"project": {"installed_capacity_mw": 50}})

    assert result["next_action"] == "ask_user_for_parameters"
    assert result["can_calculate"] is False
    assert result["follow_up_questions"]


def test_agent_tool_schema_exposes_canonical_project_fields():
    schema = validate_project_parameters_tool.args_schema.model_json_schema()
    project_schema = schema["$defs"]["PartialProjectParameters"]

    assert "installed_capacity_mw" in project_schema["properties"]
    assert "unit_capex_yuan_per_kw" in project_schema["properties"]
    assert "capacity_mw" not in project_schema["properties"]
    assert "unit_investment_yuan_per_kw" not in project_schema["properties"]
