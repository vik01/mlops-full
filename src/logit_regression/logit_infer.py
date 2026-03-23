"""
Educational Goal:
- Why this module exists in an MLOps system: Inference standardizes how we
generate predictions so downstream systems can rely on a stable schema.
- Responsibility (separation of concerns): This module only runs model.predict
    and formats outputs; it does not load data or evaluate metrics.
- Pipeline contract (inputs and outputs): Input is a trained model Pipeline
    and feature dataframe; output is a prediction dataframe with a single col.

TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

# Standard Library Imports
import logging

# Third-party Imports
import pandas as pd

logger = logging.getLogger(__name__)


def run_logit_inference(model, X_infer: pd.DataFrame, optimal_threshold: float) -> pd.DataFrame:
    """
    Inputs:
    - model: fitted sklearn Pipeline (includes preprocessing + estimator)
    - X_infer: dataframe of input features
    Outputs:
    - DataFrame with a SINGLE column named "prediction" that preserves the
    input index
    Why this contract matters for reliable ML delivery:
    - A stable prediction output reduces integration bugs and keeps
    batch/online serving consistent.
    """
    logger.info("Generating predictions")

    y_proba = model.predict_proba(X_infer)[:, 1]
    y_pred = (y_proba >= optimal_threshold).astype(int)

    result = X_infer.copy()
    result["prediction"] = y_pred
    result["predict_proba"] = y_proba

    return result
