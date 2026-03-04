"""
Module: Decision Tree Training
--------------------------------
Role: Train a Decision Tree classifier and return a fitted sklearn Pipeline.
Input: X_train, y_train (DataFrames), a preprocessor (ColumnTransformer), problem_type (str).
Output: Fitted sklearn Pipeline (preprocessor + DecisionTreeClassifier).

Educational Goal:
- Why this module exists in an MLOps system:
    Isolating model training means you can retrain without touching evaluation
    or inference logic.
- Responsibility (separation of concerns): Only build and fit the Pipeline.
    No evaluation, no saving, no plotting.
- Pipeline contract (inputs and outputs): Takes training data and returns a
    fitted Pipeline that main.py can pass to evaluate and infer.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def train_dtrees_model(X_train, y_train, preprocessor, problem_type):
    """
    Inputs:
    - X_train: Feature matrix for training (pandas DataFrame).
    - y_train: Target labels for training (pandas Series).
    - preprocessor: Unfitted ColumnTransformer from src.features.
    - problem_type: "classification" or "regression" (str).
    Outputs:
    - pipeline: Fitted sklearn Pipeline (preprocessor + DecisionTreeClassifier).
    Why this contract matters for reliable ML delivery:
    - Wrapping preprocessor and estimator in one Pipeline ensures the same
      transformations at training, evaluation, and inference — no leakage.
    """
    print("[dtrees_train.train_model] Building Decision Tree pipeline...")  # TODO: replace with logging later

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

    print(f"[dtrees_train.train_model] Fitting on {len(X_train)} samples...")  # TODO: replace with logging later

    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        raise RuntimeError(
            f"[dtrees_train.train_model] Pipeline fitting failed: {exc}"
        ) from exc

    print("[dtrees_train.train_model] Training complete.")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace or extend the baseline estimator above.
    # Why: Baseline uses fixed hyperparameters from the notebook.
    #      You may want to tune max_depth or add GridSearchCV later.
    # Examples:
    # 1. from sklearn.model_selection import GridSearchCV
    #    param_grid = {"model__max_depth": [3, 4, 5]}
    #    pipeline = GridSearchCV(pipeline, param_grid, cv=5)
    # 2. Try min_samples_split or different random_state.
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return pipeline