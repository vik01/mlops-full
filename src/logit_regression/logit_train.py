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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression


def train_logit_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocessor: ColumnTransformer,
        problem_type: str,
        random_state: int = 42,
        max_iterations: int = 500):
    """
    Inputs:
    - X_train: training features dataframe
    - y_train: training target series
    - preprocessor: sklearn ColumnTransformer recipe (not fitted yet)
    - problem_type: "regression" or "classification"
    Outputs:
    - fitted sklearn Pipeline(preprocess -> model)
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if X_train.empty:
        raise ValueError("X_train must not be empty")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")
    if y_train.empty:
        raise ValueError("y_train must not be empty")
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("preprocessor must be a ColumnTransformer")
    if not isinstance(problem_type, str):
        raise TypeError("problem_type must be a string")
    if not isinstance(random_state, int):
        raise TypeError("random_state must be an int")
    if not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an int")

    print("""[logit_train.logit_train]  Fitting model Pipeline
          (preprocess -> estimator)""")
    # TODO: replace with logging later

    if problem_type not in ["regression", "classification"]:
        raise ValueError(
            "train_model: problem_type must be "
            "regression or classification"
        )

    estimator = (
        Ridge()
        if problem_type == "regression"
        else LogisticRegression(random_state=random_state,
                                max_iter=max_iterations)
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

    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        raise RuntimeError(
            f"[logit_train.logit_train] Pipeline fitting failed: {exc}"
        ) from exc

    return model
