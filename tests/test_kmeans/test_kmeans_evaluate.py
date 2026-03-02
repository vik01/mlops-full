import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from kmeans.evaluate import evaluate_kmeans_model


def build_dummy_preprocessor():
    """
    Minimal preprocessor compatible with the KMeans pipeline.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
                ["cat_feature"],
            ),
            ("num", "passthrough", ["num_feature"]),
        ],
        remainder="drop",
    )


def build_dummy_dataframe():
    """
    Small deterministic dataset.
    """
    return pd.DataFrame(
        {
            "num_feature": [0, 1, 2, 3, 4, 5],
            "cat_feature": ["A", "B", "A", "B", "A", "B"],
        }
    )


def build_fitted_model():
    """
    Create and fit a minimal Pipeline for testing evaluation.
    """
    df = build_dummy_dataframe()
    preprocessor = build_dummy_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", KMeans(n_clusters=2, random_state=42, n_init=10)),
        ]
    )

    model.fit(df)
    return model, df


def test_evaluate_returns_float():
    """
    Test that evaluate_kmeans_model returns a float.
    """
    model, df = build_fitted_model()

    score = evaluate_kmeans_model(model, df)

    assert isinstance(score, float)


def test_evaluate_handles_small_sample():
    """
    Test guard condition when silhouette cannot be computed.
    """
    df_small = pd.DataFrame(
        {
            "num_feature": [0, 1],
            "cat_feature": ["A", "B"],
        }
    )

    preprocessor = build_dummy_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", KMeans(n_clusters=2, random_state=42, n_init=10)),
        ]
    )

    model.fit(df_small)

    score = evaluate_kmeans_model(model, df_small)

    # Guard should return 0.0
    assert score == 0.0


def test_evaluate_with_single_cluster():
    """
    Test guard condition when all samples fall into one cluster.
    """
    df = build_dummy_dataframe()
    preprocessor = build_dummy_preprocessor()

    # Force 1 cluster
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", KMeans(n_clusters=1, random_state=42, n_init=10)),
        ]
    )

    model.fit(df)

    score = evaluate_kmeans_model(model, df)

    assert score == 0.0