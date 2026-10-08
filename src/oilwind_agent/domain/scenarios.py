"""Reproducible CNY scenario simulation migrated from the thesis baseline."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from oilwind_agent.domain.economics import calculate_economic_metrics
from oilwind_agent.domain.project_parameters import ProjectParameters

RANDOM_STATE = 42
HISTORICAL_VND_PER_CNY = 3356.23


def _yuan_range(lower_vnd: float, upper_vnd: float) -> tuple[float, float]:
    """Convert historical scenario bounds once; user input remains CNY-only."""
    return lower_vnd / HISTORICAL_VND_PER_CNY, upper_vnd / HISTORICAL_VND_PER_CNY


SCENARIO_DEFINITIONS: dict[str, dict[str, Any]] = {
    "pessimistic": {
        "label": "悲观情景",
        "category": "headline",
        "full_load_hours_range": (2000.0, 2800.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1200, 1500),
        "opex_ratio_range": (0.03, 0.04),
        "carbon_price_yuan_per_ton": 0.0,
    },
    "baseline": {
        "label": "基准情景",
        "category": "headline",
        "full_load_hours_range": (2000.0, 3500.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1200, 1800),
        "opex_ratio_range": (0.02, 0.04),
        "carbon_price_yuan_per_ton": 0.0,
    },
    "optimistic": {
        "label": "乐观情景",
        "category": "headline",
        "full_load_hours_range": (2600.0, 3500.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1500, 1800),
        "opex_ratio_range": (0.02, 0.03),
        "carbon_price_yuan_per_ton": 50.0,
    },
    "tariff_low": {
        "label": "低电价",
        "category": "tariff",
        "full_load_hours_range": (2000.0, 3500.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1000, 1400),
        "opex_ratio_range": (0.02, 0.04),
        "carbon_price_yuan_per_ton": 0.0,
    },
    "tariff_high": {
        "label": "高电价",
        "category": "tariff",
        "full_load_hours_range": (2000.0, 3500.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1600, 2000),
        "opex_ratio_range": (0.02, 0.04),
        "carbon_price_yuan_per_ton": 0.0,
    },
    "full_hours_low": {
        "label": "低利用小时",
        "category": "full_hours",
        "full_load_hours_range": (1800.0, 2600.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1200, 1800),
        "opex_ratio_range": (0.02, 0.04),
        "carbon_price_yuan_per_ton": 0.0,
    },
    "full_hours_high": {
        "label": "高利用小时",
        "category": "full_hours",
        "full_load_hours_range": (2800.0, 3800.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1200, 1800),
        "opex_ratio_range": (0.02, 0.04),
        "carbon_price_yuan_per_ton": 0.0,
    },
    "opex_low": {
        "label": "低OPEX",
        "category": "opex",
        "full_load_hours_range": (2000.0, 3500.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1200, 1800),
        "opex_ratio_range": (0.015, 0.025),
        "carbon_price_yuan_per_ton": 0.0,
    },
    "opex_high": {
        "label": "高OPEX",
        "category": "opex",
        "full_load_hours_range": (2000.0, 3500.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1200, 1800),
        "opex_ratio_range": (0.035, 0.05),
        "carbon_price_yuan_per_ton": 0.0,
    },
    "carbon_50": {
        "label": "碳价50元/吨",
        "category": "carbon_price",
        "full_load_hours_range": (2000.0, 3500.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1200, 1800),
        "opex_ratio_range": (0.02, 0.04),
        "carbon_price_yuan_per_ton": 50.0,
    },
    "carbon_100": {
        "label": "碳价100元/吨",
        "category": "carbon_price",
        "full_load_hours_range": (2000.0, 3500.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1200, 1800),
        "opex_ratio_range": (0.02, 0.04),
        "carbon_price_yuan_per_ton": 100.0,
    },
    "combined_stress": {
        "label": "组合压力情景",
        "category": "combined",
        "full_load_hours_range": (1800.0, 2600.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1000, 1400),
        "opex_ratio_range": (0.035, 0.05),
        "carbon_price_yuan_per_ton": 0.0,
    },
    "combined_upside": {
        "label": "组合利好情景",
        "category": "combined",
        "full_load_hours_range": (2800.0, 3800.0),
        "tariff_yuan_per_kwh_range": _yuan_range(1600, 2000),
        "opex_ratio_range": (0.015, 0.025),
        "carbon_price_yuan_per_ton": 100.0,
    },
}


def simulate_scenarios(
    project: ProjectParameters,
    scenario_names: Sequence[str] | None = None,
    random_state: int = RANDOM_STATE,
) -> list[dict[str, Any]]:
    """Run named scenarios with the baseline's same-seed sampling method."""
    selected_names = (
        list(scenario_names) if scenario_names is not None else list(SCENARIO_DEFINITIONS)
    )
    unknown = [name for name in selected_names if name not in SCENARIO_DEFINITIONS]
    if unknown:
        raise ValueError(f"未知情景：{unknown}")

    results = []
    for name in selected_names:
        definition = SCENARIO_DEFINITIONS[name]
        rng = np.random.RandomState(random_state)
        full_hours = float(rng.uniform(*definition["full_load_hours_range"]))
        tariff = float(rng.uniform(*definition["tariff_yuan_per_kwh_range"]))
        opex = float(rng.uniform(*definition["opex_ratio_range"]))
        scenario_project = project.model_copy(
            update={
                "full_load_hours_per_year": full_hours,
                "tariff_yuan_per_kwh": tariff,
                "opex_ratio": opex,
                "carbon_price_yuan_per_ton": definition["carbon_price_yuan_per_ton"],
            }
        )
        results.append(
            {
                "scenario": name,
                "label": definition["label"],
                "category": definition["category"],
                "inputs": {
                    "full_load_hours_per_year": full_hours,
                    "tariff_yuan_per_kwh": tariff,
                    "opex_ratio": opex,
                    "carbon_price_yuan_per_ton": scenario_project.carbon_price_yuan_per_ton,
                },
                "metrics": calculate_economic_metrics(scenario_project),
            }
        )
    return results
