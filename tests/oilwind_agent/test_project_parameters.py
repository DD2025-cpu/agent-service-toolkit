import pytest
from pydantic import ValidationError

from oilwind_agent.domain import ProjectParameters

BASE_INPUT = {
    "installed_capacity_mw": 50,
    "unit_capex_yuan_per_kw": 5_200,
    "full_load_hours_per_year": 2_400,
    "tariff_yuan_per_kwh": 0.35,
    "opex_ratio": 0.03,
}


def test_unit_capex_derives_total_investment_and_exposes_defaults():
    project = ProjectParameters.model_validate(BASE_INPUT)

    assert project.currency == "CNY"
    assert project.total_investment_yuan == 260_000_000
    assert project.project_life_years == 20
    assert project.discount_rate == 0.08
    assert project.annual_degradation_rate == 0.005
    assert project.residual_value_rate == 0.05
    assert project.corporate_income_tax_rate == 0.20


def test_total_investment_derives_unit_capex():
    data = BASE_INPUT | {
        "unit_capex_yuan_per_kw": None,
        "total_investment_yuan": 260_000_000,
    }
    project = ProjectParameters.model_validate(data)

    assert project.unit_capex_yuan_per_kw == 5_200


def test_inconsistent_investment_values_are_rejected():
    data = BASE_INPUT | {"total_investment_yuan": 300_000_000}

    with pytest.raises(ValidationError, match="inconsistent"):
        ProjectParameters.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("installed_capacity_mw", 0),
        ("full_load_hours_per_year", 8_761),
        ("tariff_yuan_per_kwh", -0.01),
        ("opex_ratio", 1.01),
    ],
)
def test_parameter_ranges_are_enforced(field, value):
    with pytest.raises(ValidationError):
        ProjectParameters.model_validate(BASE_INPUT | {field: value})


def test_unknown_or_non_cny_fields_are_rejected():
    with pytest.raises(ValidationError):
        ProjectParameters.model_validate(BASE_INPUT | {"currency": "VND"})
    with pytest.raises(ValidationError):
        ProjectParameters.model_validate(BASE_INPUT | {"tariff_vnd_per_kwh": 1_500})
