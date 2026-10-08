"""Adapters and runtime loading for saved wind-project models."""

from oilwind_agent.model.artifacts import (
    DEFAULT_MODEL_NAME,
    ModelArtifactError,
    clear_model_cache,
    explain_irr,
    load_irr_model,
    predict_irr,
)
from oilwind_agent.model.feature_adapter import (
    IRR_MODEL_FEATURE_ORDER,
    IrrModelFeatures,
    adapt_project_to_irr_features,
    adapt_projects_to_irr_frame,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "IRR_MODEL_FEATURE_ORDER",
    "IrrModelFeatures",
    "ModelArtifactError",
    "adapt_project_to_irr_features",
    "adapt_projects_to_irr_frame",
    "clear_model_cache",
    "explain_irr",
    "load_irr_model",
    "predict_irr",
]
