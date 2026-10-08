import json

import pytest

from oilwind_agent.model.artifacts import load_irr_model
from oilwind_agent.tools import (
    ALL_OILWIND_TOOLS,
    calculate_economic_metrics_result,
    calculate_economic_metrics_tool,
    compare_projects_result,
    compare_wind_projects_tool,
    explain_irr_prediction_result,
    generate_project_report_result,
    generate_wind_project_report_tool,
    predict_irr_result,
    simulate_scenarios_result,
    simulate_scenarios_tool,
)

PROJECT = {
    "project_id": "A",
    "installed_capacity_mw": 50,
    "unit_capex_yuan_per_kw": 5_200,
    "full_load_hours_per_year": 2_400,
    "tariff_yuan_per_kwh": 0.35,
    "opex_ratio": 0.03,
}


def test_economic_tool_matches_thesis_baseline_for_equivalent_cny_input():
    result = calculate_economic_metrics_result(PROJECT)
    metrics = result["metrics"]

    assert result["status"] == "success"
    assert metrics["total_investment_yuan"] == 260_000_000
    assert metrics["first_year_generation_kwh"] == 120_000_000
    assert metrics["irr"] == pytest.approx(0.09243820486833565)
    assert metrics["npv_yuan"] == pytest.approx(24_349_135.52021704)
    assert metrics["lcoe_yuan_per_kwh"] == pytest.approx(0.2933753348057466)
    assert result["assumptions"]["project_life_years"] == 20


def test_business_tool_blocks_incomplete_project_with_structured_result():
    result = calculate_economic_metrics_tool.invoke({"project": {"installed_capacity_mw": 50}})

    assert result["status"] == "needs_input"
    assert result["validation"]["can_calculate"] is False
    assert "tariff_yuan_per_kwh" in result["validation"]["missing_required"]


def test_packaged_model_loads_and_prediction_is_reproducible():
    booster, metadata = load_irr_model()
    result = predict_irr_result(PROJECT)

    assert booster.feature_names == metadata["feature_columns"]
    assert metadata["model_version"] == "2.0.0"
    assert result["predicted_irr"] == pytest.approx(0.05628449097275734)
    assert result["model"]["artifact_sha256"] == metadata["model_sha256"]
    assert result["warnings"]
    assert "人民币口径代理数据集" in result["warnings"][0]
    assert "不代表基于真实中国风电项目样本训练" in result["warnings"][0]


def test_tree_shap_reconstructs_the_model_prediction():
    prediction = predict_irr_result(PROJECT)["predicted_irr"]
    result = explain_irr_prediction_result(PROJECT)
    explanation = result["explanation"]

    assert explanation["calculation_backend"] == ("xgboost_pred_contribs_exact_tree_shap")
    assert len(explanation["contributions"]) == 5
    assert explanation["reconstructed_prediction"] == pytest.approx(
        prediction,
        abs=1e-8,
    )


def test_scenarios_are_reproducible_and_cover_headline_cases():
    first = simulate_scenarios_result(
        PROJECT,
        ["pessimistic", "baseline", "optimistic"],
    )
    second = simulate_scenarios_result(
        PROJECT,
        ["pessimistic", "baseline", "optimistic"],
    )

    assert first == second
    assert first["scenario_count"] == 3
    assert first["summary"] == {
        "best_scenario_by_irr": "optimistic",
        "worst_scenario_by_irr": "pessimistic",
    }


def test_project_comparison_ranks_by_calculated_irr():
    better_project = PROJECT | {"project_id": "B", "tariff_yuan_per_kwh": 0.45}
    result = compare_projects_result([PROJECT, better_project])

    assert result["project_count"] == 2
    assert result["projects"][0]["project_id"] == "B"
    assert result["projects"][0]["rank_by_calculated_irr"] == 1


def test_report_is_saved_inside_controlled_output_directory(tmp_path, monkeypatch):
    from oilwind_agent.tools import business

    monkeypatch.setattr(business, "REPORT_DIR", tmp_path)
    result = generate_project_report_result(PROJECT, include_scenarios=False)
    report_path = tmp_path / "A_report.md"

    assert result["status"] == "success"
    assert report_path.is_file()
    assert "## 经济指标" in report_path.read_text(encoding="utf-8")
    assert "TreeSHAP" not in result["report_markdown"]


def test_all_registered_tool_results_are_json_serializable():
    result = {
        "economic": calculate_economic_metrics_result(PROJECT),
        "prediction": predict_irr_result(PROJECT),
        "explanation": explain_irr_prediction_result(PROJECT),
        "scenarios": simulate_scenarios_result(PROJECT, ["baseline"]),
    }

    json.dumps(result, ensure_ascii=False, allow_nan=False)
    assert [item.name for item in ALL_OILWIND_TOOLS] == [
        "validate_project_parameters",
        "calculate_economic_metrics",
        "predict_irr",
        "explain_irr_prediction",
        "simulate_scenarios",
        "compare_wind_projects",
        "generate_wind_project_report",
    ]


def test_langchain_wrappers_return_structured_results(tmp_path, monkeypatch):
    from oilwind_agent.tools import business

    monkeypatch.setattr(business, "REPORT_DIR", tmp_path)
    scenario = simulate_scenarios_tool.invoke({"project": PROJECT, "scenario_names": ["baseline"]})
    comparison = compare_wind_projects_tool.invoke(
        {
            "projects": [
                PROJECT,
                PROJECT | {"project_id": "B", "tariff_yuan_per_kwh": 0.45},
            ]
        }
    )
    report = generate_wind_project_report_tool.invoke(
        {"project": PROJECT, "include_scenarios": False}
    )

    assert scenario["status"] == "success"
    assert comparison["status"] == "success"
    assert report["status"] == "success"
    assert (tmp_path / "A_report.md").is_file()
