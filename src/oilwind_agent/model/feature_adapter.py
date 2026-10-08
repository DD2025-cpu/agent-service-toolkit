"""Translate canonical CNY project input into the saved IRR model schema."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from oilwind_agent.domain import ProjectParameters

IRR_MODEL_FEATURE_ORDER = (
    "capacity_mw",
    "investment_million_cny",
    "full_hours",
    "tariff_cny_per_kwh",
    "opex_ratio",
)


class IrrModelFeatures(BaseModel):
    """The exact ordered feature contract used by the saved XGBoost v2 model."""

    model_config = ConfigDict(extra="forbid")

    capacity_mw: float
    investment_million_cny: float
    full_hours: float
    tariff_cny_per_kwh: float
    opex_ratio: float

    def as_ordered_dict(self) -> dict[str, float]:
        values = self.model_dump()
        return {name: values[name] for name in IRR_MODEL_FEATURE_ORDER}


def _validate_project(
    project: ProjectParameters | Mapping[str, Any],
) -> ProjectParameters:
    if isinstance(project, ProjectParameters):
        return project
    return ProjectParameters.model_validate(project)


def adapt_project_to_irr_features(
    project: ProjectParameters | Mapping[str, Any],
) -> IrrModelFeatures:
    """Adapt one CNY project without exchange-rate conversion.

    The current v2 model was trained with CNY investment and tariff features.
    Only field names and total-investment scale are converted here.
    """
    normalized = _validate_project(project)
    total_investment_yuan = normalized.total_investment_yuan
    if total_investment_yuan is None:
        raise ValueError("validated project has no total investment")
    return IrrModelFeatures(
        capacity_mw=normalized.installed_capacity_mw,
        investment_million_cny=total_investment_yuan / 1_000_000,
        full_hours=normalized.full_load_hours_per_year,
        tariff_cny_per_kwh=normalized.tariff_yuan_per_kwh,
        opex_ratio=normalized.opex_ratio,
    )


def adapt_projects_to_irr_frame(
    projects: Sequence[ProjectParameters | Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a prediction frame with the training feature order preserved."""
    rows = [adapt_project_to_irr_features(project).as_ordered_dict() for project in projects]
    return pd.DataFrame(rows, columns=IRR_MODEL_FEATURE_ORDER, dtype=float)
