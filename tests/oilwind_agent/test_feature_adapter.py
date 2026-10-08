from oilwind_agent.model import (
    IRR_MODEL_FEATURE_ORDER,
    adapt_project_to_irr_features,
    adapt_projects_to_irr_frame,
)

PROJECT = {
    "installed_capacity_mw": 50,
    "total_investment_yuan": 260_000_000,
    "full_load_hours_per_year": 2_400,
    "tariff_yuan_per_kwh": 0.35,
    "opex_ratio": 0.03,
}


def test_adapter_matches_saved_v2_model_contract():
    features = adapt_project_to_irr_features(PROJECT)

    assert list(features.as_ordered_dict()) == list(IRR_MODEL_FEATURE_ORDER)
    assert features.as_ordered_dict() == {
        "capacity_mw": 50.0,
        "investment_million_cny": 260.0,
        "full_hours": 2_400.0,
        "tariff_cny_per_kwh": 0.35,
        "opex_ratio": 0.03,
    }


def test_batch_adapter_preserves_feature_order_and_numeric_types():
    frame = adapt_projects_to_irr_frame([PROJECT, PROJECT | {"project_id": "B"}])

    assert list(frame.columns) == list(IRR_MODEL_FEATURE_ORDER)
    assert frame.shape == (2, 5)
    assert all(dtype.kind == "f" for dtype in frame.dtypes)
