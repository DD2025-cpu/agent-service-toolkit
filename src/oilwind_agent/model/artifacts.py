"""Validated loading, prediction and TreeSHAP for packaged XGBoost models."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from oilwind_agent.model.feature_adapter import IRR_MODEL_FEATURE_ORDER, IrrModelFeatures

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_REGISTRY_PATH = ARTIFACT_DIR / "model_registry.json"
SUPPORTED_ARTIFACT_SCHEMA_VERSION = 1
SUPPORTED_MODEL_VERSION = "2.0.0"
DEFAULT_MODEL_NAME = "baseline_xgb"


class ModelArtifactError(ValueError):
    """A packaged model or its metadata does not satisfy the runtime contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelArtifactError(f"无法读取模型元数据：{path.name}") from error


@lru_cache(maxsize=4)
def load_irr_model(model_name: str = DEFAULT_MODEL_NAME) -> tuple[xgb.Booster, dict[str, Any]]:
    """Load a registered model after schema, feature and checksum verification."""
    registry = _read_json(MODEL_REGISTRY_PATH)
    if registry.get("schema_version") != SUPPORTED_ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError("模型注册表结构版本不受支持。")
    artifacts = registry.get("artifacts", {})
    if model_name not in artifacts:
        raise ModelArtifactError(f"未知模型：{model_name}")

    metadata_name = Path(artifacts[model_name])
    if metadata_name.name != str(metadata_name):
        raise ModelArtifactError("模型元数据必须位于工件目录。")
    metadata = _read_json(ARTIFACT_DIR / metadata_name)
    if metadata.get("schema_version") != SUPPORTED_ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError("模型工件结构版本不受支持。")
    if metadata.get("model_version") != SUPPORTED_MODEL_VERSION:
        raise ModelArtifactError("模型版本与Agent运行时不匹配。")
    if metadata.get("feature_columns") != list(IRR_MODEL_FEATURE_ORDER):
        raise ModelArtifactError("模型特征名称或顺序与Agent适配器不一致。")

    model_name_path = Path(metadata.get("model_file", ""))
    if model_name_path.name != str(model_name_path):
        raise ModelArtifactError("模型文件必须位于工件目录。")
    model_path = ARTIFACT_DIR / model_name_path
    if not model_path.is_file() or _sha256(model_path) != metadata.get("model_sha256"):
        raise ModelArtifactError("模型文件缺失或SHA-256校验失败。")

    booster = xgb.Booster()
    booster.load_model(model_path)
    if booster.feature_names != list(IRR_MODEL_FEATURE_ORDER):
        raise ModelArtifactError("模型内部特征顺序与元数据不一致。")
    return booster, metadata


def _feature_frame(features: IrrModelFeatures) -> pd.DataFrame:
    return pd.DataFrame(
        [features.as_ordered_dict()],
        columns=IRR_MODEL_FEATURE_ORDER,
        dtype=float,
    )


def predict_irr(
    features: IrrModelFeatures, model_name: str = DEFAULT_MODEL_NAME
) -> tuple[float, dict[str, Any]]:
    """Predict one project's IRR with the packaged model."""
    booster, metadata = load_irr_model(model_name)
    frame = _feature_frame(features)
    matrix = xgb.DMatrix(frame, feature_names=list(IRR_MODEL_FEATURE_ORDER))
    prediction = float(booster.predict(matrix, validate_features=True)[0])
    return prediction, metadata


def explain_irr(
    features: IrrModelFeatures,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return exact per-feature TreeSHAP contributions for one prediction."""
    booster, metadata = load_irr_model(model_name)
    frame = _feature_frame(features)
    matrix = xgb.DMatrix(frame, feature_names=list(IRR_MODEL_FEATURE_ORDER))
    raw = np.asarray(
        booster.predict(matrix, pred_contribs=True, validate_features=True)[0],
        dtype=float,
    )
    shap_values = raw[:-1]
    base_value = float(raw[-1])
    prediction = float(base_value + shap_values.sum())
    contributions = [
        {
            "feature": feature,
            "feature_value": float(frame.iloc[0][feature]),
            "shap_value": float(shap_values[position]),
            "effect": (
                "increase"
                if shap_values[position] > 0
                else "decrease"
                if shap_values[position] < 0
                else "neutral"
            ),
        }
        for position, feature in enumerate(IRR_MODEL_FEATURE_ORDER)
    ]
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return (
        {
            "base_value": base_value,
            "predicted_irr": prediction,
            "reconstructed_prediction": prediction,
            "contributions": contributions,
            "calculation_backend": "xgboost_pred_contribs_exact_tree_shap",
            "interpretation": (
                "正SHAP值提高预测IRR，负SHAP值降低预测IRR；SHAP反映模型关系，不代表因果关系。"
            ),
        },
        metadata,
    )


def clear_model_cache() -> None:
    """Clear loaded model objects after an intentional artifact update."""
    load_irr_model.cache_clear()
