"""
Educational Goal:
- Why this module exists in an MLOps system: Inference standardizes how we
generate predictions so downstream systems can rely on a stable schema.
- Responsibility (separation of concerns): This module only runs model.predict
    and formats outputs; it does not load data or evaluate metrics.
- Pipeline contract (inputs and outputs): Input is a trained model Pipeline
    and feature dataframe; output is a prediction dataframe with a single col.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

import pandas as pd


def run_logit_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
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
    print("run_inference: generating predictions")
    # TODO: replace with logging later

    preds = model.predict(X_infer)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend
    # Why: Some products require probabilities, thresholds,
    # or post-processing rules.
    # Examples:
    # 1. For classification, return probabilities via model.predict_proba
    # (but keep output format stable)
    # 2. Apply a business threshold to convert probabilities to labels
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to
    # proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return pd.DataFrame({"prediction": preds}, index=X_infer.index)
