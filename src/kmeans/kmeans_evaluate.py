"""
Educational Goal:
- Why this module exists in an MLOps system: Provide clustering metric.
- Responsibility (separation of concerns): Compute silhouette score.
- Pipeline contract (inputs and outputs): model + X_test -> float.

TODO: Replace print statements with standard library logging in a later session
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


def evaluate_kmeans_model(model: Pipeline, X_test: pd.DataFrame) -> float:
    """
    Inputs:
    - model: Fitted Pipeline(preprocess + KMeans).
    - X_test: Test features.
    Outputs:
    - float silhouette score.
    Why this contract matters for reliable ML delivery:
    - Provides a quantitative signal of clustering quality.
    """

    # TODO: replace with logging later
    print("[kmeans.evaluate] Evaluating silhouette score")

    X_trans = model.named_steps["preprocess"].transform(X_test)
    labels = model.named_steps["model"].predict(X_trans)

    # Guard against invalid silhouette conditions
    if len(set(labels)) < 2 or len(X_trans) < 3:
        print("Not enough samples or clusters for silhouette. Returning 0.0")
        return 0.0

    # ============================================================
    # FULL SILHOUETTE (exact calculation, no sampling)
    # ============================================================
    score = float(silhouette_score(X_trans, labels))

    # ============================================================
    # OPTIONAL: SAMPLE-BASED SILHOUETTE (for large datasets)
    # Uncomment if computation becomes too slow.
    # ============================================================
    #
    # score = float(
    #     silhouette_score(
    #         X_trans,
    #         labels,
    #         sample_size=min(1000, len(X_trans)),
    #         random_state=42
    #     )
    # )
    #
    # Why sampling:
    # Silhouette has O(n^2) complexity.
    # For large datasets, sampling provides a scalable approximation.
    #

    return score
