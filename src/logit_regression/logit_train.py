"""
Educational Goal:
- Why this module exists in an MLOps system: Training is separated so it can
    be reused, tested, and swapped without rewriting the pipeline.
- Responsibility (separation of concerns): This module fits a single sklearn
    Pipeline that bundles preprocessing + model.
- Pipeline contract (inputs and outputs): Inputs are X_train, y_train, a
    preprocessor recipe, and problem_type; output is a fitted sklearn Pipeline.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression


def train_logit_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocessor,
        problem_type: str):
    """
    Inputs:
    - X_train: training features dataframe
    - y_train: training target series
    - preprocessor: sklearn ColumnTransformer recipe (not fitted yet)
    - problem_type: "regression" or "classification"
    Outputs:
    - fitted sklearn Pipeline(preprocess -> model)
    Why this contract matters for reliable ML delivery:
    - Putting preprocessing + model in one Pipeline prevents training/serving ç
    skew and reduces leakage risk.
    """
    print("train_model: fitting model Pipeline (preprocess -> estimator)")
    # TODO: replace with logging later

    if problem_type not in ["regression", "classification"]:
        raise ValueError(
            "train_model: problem_type must be "
            "regression or classification"
        )

    estimator = (
        Ridge()
        if problem_type == "regression"
        else LogisticRegression(max_iter=500)
    )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend
    # Why: Hyperparameters depend on dataset size, imbalance, and business
    # goals.
    # Examples:
    # 1. LogisticRegression(class_weight="balanced")
    # 2. LogisticRegression(C=0.5, solver="liblinear")
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

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    model.fit(X_train, y_train)
    return model
