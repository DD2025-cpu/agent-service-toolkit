"""Canonical CNY input model for a wind-power project."""

import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveCapacityMw = Annotated[float, Field(gt=0, le=10_000)]
PositiveInvestmentYuan = Annotated[float, Field(gt=0, le=1_000_000_000_000)]
PositiveUnitCapex = Annotated[float, Field(gt=0, le=100_000)]
FullLoadHours = Annotated[float, Field(gt=0, le=8_760)]
TariffYuanPerKwh = Annotated[float, Field(ge=0, le=10)]
Ratio = Annotated[float, Field(ge=0, le=1)]


class ProjectParameters(BaseModel):
    """Validated project input used by every OilWind-Agent business tool.

    External monetary inputs are always expressed in CNY. Callers may provide
    either total investment or unit CAPEX; validation derives the other value.
    Project-specific operating inputs remain required so the Agent cannot hide
    missing facts behind defaults.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    currency: Literal["CNY"] = Field(
        default="CNY",
        description="Currency for every monetary input.",
    )
    project_id: str | None = Field(default=None, min_length=1, max_length=100)
    province: str | None = Field(default=None, min_length=1, max_length=100)

    installed_capacity_mw: PositiveCapacityMw = Field(
        description="Installed capacity.",
        json_schema_extra={"unit": "MW"},
    )
    total_investment_yuan: PositiveInvestmentYuan | None = Field(
        default=None,
        description="Total project investment in CNY.",
        json_schema_extra={"unit": "CNY"},
    )
    unit_capex_yuan_per_kw: PositiveUnitCapex | None = Field(
        default=None,
        description="Investment per kW in CNY.",
        json_schema_extra={"unit": "CNY/kW"},
    )
    full_load_hours_per_year: FullLoadHours = Field(
        description="Annual equivalent full-load hours.",
        json_schema_extra={"unit": "h/year"},
    )
    tariff_yuan_per_kwh: TariffYuanPerKwh = Field(
        description="On-grid electricity tariff in CNY per kWh.",
        json_schema_extra={"unit": "CNY/kWh"},
    )
    opex_ratio: Ratio = Field(
        description="Annual operating cost as a share of total investment.",
        json_schema_extra={"unit": "ratio"},
    )

    project_life_years: int = Field(default=20, ge=1, le=50)
    discount_rate: Ratio = 0.08
    annual_degradation_rate: Annotated[float, Field(ge=0, lt=1)] = 0.005
    residual_value_rate: Ratio = 0.05
    corporate_income_tax_rate: Ratio = 0.20
    other_revenue_tax_rate: Ratio = 0.0
    carbon_price_yuan_per_ton: Annotated[float, Field(ge=0, le=10_000)] = 0.0
    emission_reduction_t_per_mwh: Annotated[float, Field(ge=0, le=10)] = 0.50

    @model_validator(mode="after")
    def normalize_investment(self) -> Self:
        """Require one investment representation and materialize both."""
        capacity_kw = self.installed_capacity_mw * 1_000
        total = self.total_investment_yuan
        unit = self.unit_capex_yuan_per_kw

        if total is None and unit is None:
            raise ValueError(
                "total_investment_yuan and unit_capex_yuan_per_kw: "
                "provide at least one investment value"
            )
        if total is None:
            self.total_investment_yuan = capacity_kw * unit  # type: ignore[operator]
        elif unit is None:
            self.unit_capex_yuan_per_kw = total / capacity_kw
        else:
            derived_total = capacity_kw * unit
            if not math.isclose(total, derived_total, rel_tol=1e-6, abs_tol=1.0):
                raise ValueError(
                    "total_investment_yuan is inconsistent with "
                    "installed_capacity_mw * 1000 * unit_capex_yuan_per_kw"
                )
        return self
