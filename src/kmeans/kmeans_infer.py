"""
Educational Goal:
- Why this module exists in an MLOps system: Standardize cluster
  assignment.
- Responsibility (separation of concerns): Predict cluster_id.
- Pipeline contract (inputs and outputs): model + X -> DataFrame with
  one column 'prediction'.

"""

# Standard Library Imports
import logging

# Third-party Imports
import pandas as pd

logger = logging.getLogger(__name__)


def run_kmeans_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: Fitted Pipeline.
    - X_infer: Features.
    Outputs:
    - DataFrame with single column 'prediction'.
    Why this contract matters for reliable ML delivery:
    - Stable output schema prevents downstream breakage.
    """
    logger.info("Predicting cluster_id")

    preds = model.predict(X_infer)
    return pd.DataFrame({"prediction": preds}, index=X_infer.index)
