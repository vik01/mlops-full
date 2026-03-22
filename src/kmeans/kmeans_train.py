"""
Educational Goal:
- Why this module exists in an MLOps system: Isolate clustering training logic.
- Responsibility (separation of concerns): Fit Pipeline(preprocess + KMeans).
- Pipeline contract (inputs and outputs): X_train -> fitted Pipeline.

"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


def train_kmeans_model(X_train: pd.DataFrame,
                       preprocessor: ColumnTransformer, n_clusters: int):
    """
    Inputs:
    - X_train: Training features.
    - preprocessor: ColumnTransformer.
    - n_clusters: Number of clusters.
    Outputs:
    - model: Fitted Pipeline.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if X_train.empty:
        raise ValueError("X_train must not be empty")
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("preprocessor must be a ColumnTransformer")
    if not isinstance(n_clusters, int):
        raise TypeError("n_clusters must be an int")

    logger.info("Training KMeans Pipeline")

    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])

    try:
        model.fit(X_train)
    except Exception as exc:
        raise RuntimeError(
            f"[kmeans.train] Pipeline fitting failed: {exc}"
        ) from exc

    return model
