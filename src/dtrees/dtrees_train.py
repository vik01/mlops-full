"""
Module: Decision Tree Training
--------------------------------
Role: Train a Decision Tree classifier and return a fitted sklearn Pipeline.
Input: X_train, y_train (DataFrames), a preprocessor (ColumnTransformer),
problem_type (str).
Output: Fitted sklearn Pipeline (preprocessor + DecisionTreeClassifier).

Educational Goal:
- Why this module exists in an MLOps system:
    Isolating model training means you can retrain without touching evaluation
    or inference logic.
- Responsibility (separation of concerns): Only build and fit the Pipeline.
    No evaluation, no saving, no plotting.
- Pipeline contract (inputs and outputs): Takes training data and returns a
    fitted Pipeline that main.py can pass to evaluate and infer.

TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def train_dtrees_model(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       preprocessor: ColumnTransformer,
                       problem_type: str):
    """
    Inputs:
    - X_train: Feature matrix for training (pandas DataFrame).
    - y_train: Target labels for training (pandas Series).
    - preprocessor: Unfitted ColumnTransformer from src.features.
    - problem_type: "classification" or "regression" (str).
    Outputs:
    - pipeline: Fitted sklearn Pipeline (preprocessor + DecisionTreeClassifier)
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

    logger.info("Building Decision Tree pipeline...")

    if problem_type not in ["regression", "classification"]:
        raise ValueError(
            "train_model: problem_type must be "
            "regression or classification"
        )

    # max_depth=4 and min_samples_leaf=30 from notebook Section 6.1.
    # These values balance interpretability and accuracy (~85%).
    estimator = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=30,
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])

    logger.info("Fitting on %d samples...", len(X_train))

    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        raise RuntimeError(
            f"[dtrees_train.train_model] Pipeline fitting failed: {exc}"
        ) from exc

    logger.info("Training complete.")

    return pipeline
