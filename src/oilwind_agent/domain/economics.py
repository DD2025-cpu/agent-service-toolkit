"""Deterministic CNY cash-flow calculations for wind projects."""

import math
from typing import Any

import numpy as np
import numpy_financial as npf

from oilwind_agent.domain.project_parameters import ProjectParameters

ECONOMIC_CALCULATION_VERSION = "cny-after-tax-v1"


def _payback_period(cashflows: np.ndarray, final_lump_sum: float = 0.0) -> float | None:
    cumulative = float(cashflows[0])
    if cumulative >= 0:
        return 0.0
    for year, cashflow in enumerate(cashflows[1:], start=1):
        previous = cumulative
        cumulative += float(cashflow)
        if cumulative >= 0 and cashflow > 0:
            return float((year - 1) + (-previous / float(cashflow)))
    if cumulative + final_lump_sum >= 0:
        return float(len(cashflows) - 1)
    return None


def economic_assumptions(project: ProjectParameters) -> dict[str, Any]:
    """Return every assumption used by the calculation."""
    return {
        "version": ECONOMIC_CALCULATION_VERSION,
        "currency": "CNY",
        "project_life_years": project.project_life_years,
        "discount_rate": project.discount_rate,
        "annual_degradation_rate": project.annual_degradation_rate,
        "tariff_escalation_rate": 0.0,
        "opex_escalation_rate": 0.0,
        "residual_value_rate": project.residual_value_rate,
        "corporate_income_tax_rate": project.corporate_income_tax_rate,
        "other_revenue_tax_rate": project.other_revenue_tax_rate,
        "carbon_price_yuan_per_ton": project.carbon_price_yuan_per_ton,
        "emission_reduction_t_per_mwh": project.emission_reduction_t_per_mwh,
        "depreciation_method": "straight_line_to_salvage",
    }


def calculate_economic_metrics(project: ProjectParameters) -> dict[str, float | None]:
    """Calculate IRR, NPV, payback periods and LCOE from validated CNY input."""
    total_investment = project.total_investment_yuan
    if total_investment is None:
        raise ValueError("validated project has no total investment")
    unit_capex = project.unit_capex_yuan_per_kw
    if unit_capex is None:
        raise ValueError("validated project has no unit CAPEX")

    years = project.project_life_years
    operating_years = np.arange(years, dtype=float)
    discount_years = np.arange(1, years + 1, dtype=float)
    generation_kwh = (
        project.installed_capacity_mw
        * 1_000
        * project.full_load_hours_per_year
        * (1 - project.annual_degradation_rate) ** operating_years
    )
    electricity_revenue = generation_kwh * project.tariff_yuan_per_kwh
    carbon_revenue = (
        generation_kwh
        / 1_000
        * project.emission_reduction_t_per_mwh
        * project.carbon_price_yuan_per_ton
    )
    revenue = electricity_revenue + carbon_revenue
    opex = np.full(years, total_investment * project.opex_ratio, dtype=float)
    other_revenue_tax = revenue * project.other_revenue_tax_rate

    salvage_value = total_investment * project.residual_value_rate
    annual_depreciation = (total_investment - salvage_value) / years
    taxable_income = revenue - opex - other_revenue_tax - annual_depreciation
    corporate_income_tax = np.maximum(taxable_income, 0) * project.corporate_income_tax_rate
    tax_and_fees = other_revenue_tax + corporate_income_tax
    operating_cashflows = revenue - opex - tax_and_fees

    operating_only = np.concatenate(([-total_investment], operating_cashflows))
    cashflows = operating_only.copy()
    cashflows[-1] += salvage_value
    discount_factors = (1 + project.discount_rate) ** np.concatenate(([0.0], discount_years))
    discounted_cashflows = cashflows / discount_factors

    irr_value = float(npf.irr(cashflows))
    irr = irr_value if math.isfinite(irr_value) else None
    npv_yuan = float(discounted_cashflows.sum())
    static_payback = _payback_period(operating_only, final_lump_sum=salvage_value)
    discounted_operating = operating_only / discount_factors
    discounted_payback = _payback_period(
        discounted_operating,
        final_lump_sum=salvage_value / discount_factors[-1],
    )

    discounted_generation = generation_kwh / ((1 + project.discount_rate) ** discount_years)
    discounted_costs = (
        total_investment
        + float(np.sum((opex + other_revenue_tax) / discount_factors[1:]))
        - salvage_value / discount_factors[-1]
    )
    lcoe = float(discounted_costs / discounted_generation.sum())

    return {
        "total_investment_yuan": float(total_investment),
        "unit_capex_yuan_per_kw": float(unit_capex),
        "first_year_generation_kwh": float(generation_kwh[0]),
        "first_year_generation_mwh": float(generation_kwh[0] / 1_000),
        "first_year_electricity_revenue_yuan": float(electricity_revenue[0]),
        "first_year_carbon_revenue_yuan": float(carbon_revenue[0]),
        "first_year_total_revenue_yuan": float(revenue[0]),
        "first_year_opex_yuan": float(opex[0]),
        "first_year_tax_and_fees_yuan": float(tax_and_fees[0]),
        "first_year_net_cashflow_yuan": float(operating_cashflows[0]),
        "salvage_value_yuan": float(salvage_value),
        "irr": irr,
        "irr_percent": None if irr is None else irr * 100,
        "npv_yuan": npv_yuan,
        "static_payback_years": static_payback,
        "discounted_payback_years": discounted_payback,
        "lcoe_yuan_per_kwh": lcoe,
    }
