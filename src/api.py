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
from typing import List

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import sys

# Add the src directory to Python path so that main.py can keep using
# imports like: from logger import ...
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import load_config
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
class HeartDiseaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Age: int
    Sex: int
    chest_pain_type: int = Field(alias="Chest pain type")
    BP: int
    Cholesterol: int
    fbs_over_120: int = Field(alias="FBS over 120")
    ekg_results: int = Field(alias="EKG results")
    max_hr: int = Field(alias="Max HR")
    exercise_angina: int = Field(alias="Exercise angina")
    st_depression: float = Field(alias="ST depression")
    slope_of_st: int = Field(alias="Slope of ST")
    number_of_vessels_fluro: int = Field(alias="Number of vessels fluro")
    Thallium: int


class PredictionRequest(BaseModel):
    data: List[HeartDiseaseRecord]


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]


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
        raise RuntimeError("Missing WANDB_ENTITY environment variable.")

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
    """Return the decision threshold used for logistic regression inference."""
    return float(cfg.get("inference", {}).get("threshold", 0.5))


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
        if not request.data:
            raise ValueError("Input data is empty.")

        df = pd.DataFrame([r.model_dump(by_alias=True) for r in request.data])

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