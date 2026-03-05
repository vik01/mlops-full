"""
Educational Goal:
- Why this module exists in an MLOps system: Evaluation provides a
  consistent, testable way to compare model iterations.
- Responsibility (separation of concerns): This module only computes a
  metric; it does not train models or save artifacts.
- Pipeline contract (inputs and outputs): Input is a fitted model Pipeline
  and held-out test set; output is a single float metric.

TODO: Replace print statements with standard library logging in a later
session
TODO: Any temporary or hardcoded variable or parameter will be imported
from config.yml in a later session
"""

import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score


def evaluate_logit_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str
) -> float:
    """
    Inputs:
    - model: fitted sklearn Pipeline
    - X_test: test features dataframe
    - y_test: test target series
    - problem_type: "regression" or "classification"
    Outputs:
    - float metric (RMSE for regression, weighted F1 for classification)
    Why this contract matters for reliable ML delivery:
    - A single float output is easy to log, compare across runs, and
      use for CI gating later.
    """
    print("evaluate_model: evaluating model on held-out test set")
    # TODO: logging later

    if problem_type not in ["regression", "classification"]:
        msg = (
            'evaluate_model: problem_type must be '
            '"regression" or "classification"'
        )
        raise ValueError(msg)

    y_pred = model.predict(X_test)

    if problem_type == "regression":
        metric_value = float(mean_squared_error(y_test, y_pred, squared=False))
    else:
        metric_value = float(f1_score(y_test, y_pred, average="weighted"))

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the
    # baseline
    # Why: Metric choice and thresholding vary by business context.
    # Examples:
    # 1. Add ROC-AUC using predict_proba for binary classification
    # 2. Print confusion matrix / classification report (prints only)
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

    return metric_value
