"""
Educational Goal:
- Why this module exists in an MLOps system: Standardize cluster assignment.
- Responsibility (separation of concerns): Predict cluster_id.
- Pipeline contract (inputs and outputs): model + X -> DataFrame with one column 'prediction'.

TODO: Replace print statements with standard library logging in a later session
"""

import pandas as pd


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
    print("[kmeans.infer] Predicting cluster_id")  # TODO

    preds = model.predict(X_infer)
    return pd.DataFrame({"prediction": preds}, index=X_infer.index)