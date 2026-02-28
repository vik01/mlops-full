"""
Educational Goal:
- Why this module exists in an MLOps system: Isolate clustering training logic.
- Responsibility (separation of concerns): Fit Pipeline(preprocess + KMeans).
- Pipeline contract (inputs and outputs): X_train -> fitted Pipeline.

TODO: Replace print statements with standard library logging in a later session
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


def train_kmeans_model(X_train: pd.DataFrame, preprocessor, n_clusters: int):
    """
    Inputs:
    - X_train: Training features.
    - preprocessor: ColumnTransformer.
    - n_clusters: Number of clusters.
    Outputs:
    - model: Fitted Pipeline.
    Why this contract matters for reliable ML delivery:
    - Bundles preprocessing and model to prevent training-serving skew.
    """
    print("[kmeans.train] Training KMeans Pipeline")  # TODO

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])

    model.fit(X_train)
    return model