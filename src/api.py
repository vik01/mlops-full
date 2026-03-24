"""
API layer for serving predictions.

Educational Goal:
- Provide a clean separation between model logic and serving layer
- Reuse existing pipeline functions (no new ML logic here)
- Expose endpoints for health check and prediction
- Load the production model from the W&B model registry using the `prod` alias

Endpoints:
- GET /health -> service status
- POST /predict -> returns predictions for input records
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys

# Add the src directory to Python path so that main.py can keep using
# imports like: from logger import ...
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import load_config
from logger import configure_logging
from logit_regression.logit_infer import run_logit_inference

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="Heart Disease Prediction API")
logger = logging.getLogger(__name__)

cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")

# -----------------------------
# Logging setup
# -----------------------------
log_level = cfg.get("logging", {}).get("level", "INFO")
log_file = cfg.get("paths", {}).get("log_file", "logs/pipeline.log")
configure_logging(log_level=log_level, log_file=log_file)


# -----------------------------
# Pydantic schemas
# -----------------------------
class PredictionRequest(BaseModel):
    data: List[dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]


# -----------------------------
# Helper functions
# -----------------------------
def get_required_feature_columns() -> list[str]:
    """
    Collect all feature columns required for inference from config.yaml.
    """
    features_cfg = cfg.get("features", {}) or {}

    required_columns = (
        list(features_cfg.get("quantile_bin", []))
        + list(features_cfg.get("categorical_onehot", []))
        + list(features_cfg.get("numeric_passthrough", []))
        + list(features_cfg.get("ordinal_encode", []))
        + list(features_cfg.get("min_max_scaler", []))
    )

    # Remove duplicates while preserving order
    return list(dict.fromkeys(required_columns))


def validate_inference_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Perform lightweight validation for inference inputs.

    This validation is intentionally inference-specific:
    - input must not be empty
    - all required feature columns must be present
    - target column is NOT required
    """
    if df.empty:
        raise ValueError("Input data is empty.")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required inference columns: {missing}")


def _load_model_from_wandb_prod():
    """
    Load the production model artifact from Weights & Biases using alias `prod`.
    """
    load_dotenv()

    wandb_cfg = cfg.get("wandb", {}) or {}
    if not wandb_cfg.get("enabled", False):
        raise RuntimeError("W&B is disabled in config. Cannot load production model.")

    if wandb is None:
        raise RuntimeError("wandb is not installed. Cannot load production model.")

    api_key = os.getenv("WANDB_API_KEY")
    entity = os.getenv("WANDB_ENTITY")
    project = wandb_cfg.get("project")
    artifact_name = wandb_cfg.get("model_artifact_name")

    if not api_key:
        raise RuntimeError("Missing WANDB_API_KEY in environment variables.")

    if not entity:
        raise RuntimeError("Missing wandb.entity in config.yaml.")

    if not project:
        raise RuntimeError("Missing wandb.project in config.yaml.")

    if not artifact_name:
        raise RuntimeError("Missing wandb.model_artifact_name in config.yaml.")

    os.environ.setdefault("WANDB_API_KEY", api_key)

    artifact_ref = f"{entity}/{project}/{artifact_name}:prod"
    logger.info("Loading production model from W&B artifact %s", artifact_ref)

    try:
        api = wandb.Api()
        artifact = api.artifact(artifact_ref, type="model")
        artifact_dir = Path(artifact.download())

        model_file = None
        for file_name in os.listdir(artifact_dir):
            if file_name.endswith(".joblib") or file_name.endswith(".pkl"):
                model_file = artifact_dir / file_name
                break

        if model_file is None:
            raise RuntimeError(
                f"No .joblib or .pkl model file found in downloaded artifact: {artifact_dir}"
            )

        model = joblib.load(model_file)
        logger.info("Production model loaded successfully from W&B.")
        return model

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load production model from W&B artifact {artifact_ref}: {exc}"
        ) from exc


def _get_optimal_threshold() -> float:
    """
    Return the decision threshold used for logistic regression inference.

    Update this if your team stored a different optimal threshold during training.
    """
    return 0.5


# -----------------------------
# Load production model at startup
# -----------------------------
try:
    model = _load_model_from_wandb_prod()
    optimal_threshold = _get_optimal_threshold()
except Exception as exc:
    logger.error("API startup failed while loading model: %s", exc)
    raise


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        df = pd.DataFrame(request.data)

        required_columns = get_required_feature_columns()
        validate_inference_dataframe(df, required_columns)

        preds_df = run_logit_inference(model, df, optimal_threshold)

        if "prediction" not in preds_df.columns:
            raise ValueError("Inference output does not contain 'prediction' column.")

        if "predict_proba" not in preds_df.columns:
            raise ValueError("Inference output does not contain 'predict_proba' column.")

        return PredictionResponse(
            predictions=preds_df["prediction"].astype(int).tolist(),
            probabilities=preds_df["predict_proba"].astype(float).tolist(),
        )

    except ValueError as exc:
        logger.error("Prediction validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Prediction failed due to an unexpected error.")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc