import pandas as pd
import numpy as np
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from kmeans.kmeans_train import train_kmeans_model


def build_dummy_preprocessor():
    """
    Build a minimal ColumnTransformer compatible with train_kmeans_model.
    """
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                ohe,
                ["cat_feature"],
            ),
            ("num", "passthrough", ["num_feature"]),
        ],
        remainder="drop",
    )


def build_dummy_dataframe():
    """
    Create a tiny deterministic dataset for testing.
    """
    return pd.DataFrame(
        {
            "num_feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "cat_feature": ["A", "B", "A", "B", "A", "B"],
        }
    )


def test_train_kmeans_returns_pipeline():
    """
    Test that the function returns a fitted sklearn Pipeline.
    """
    df = build_dummy_dataframe()
    preprocessor = build_dummy_preprocessor()

    model = train_kmeans_model(
        X_train=df,
        preprocessor=preprocessor,
        n_clusters=2,
    )

    assert isinstance(model, Pipeline)
    assert "preprocess" in model.named_steps
    assert "model" in model.named_steps


def test_kmeans_model_is_fitted():
    """
    Test that the returned model is fitted and can predict.
    """
    df = build_dummy_dataframe()
    preprocessor = build_dummy_preprocessor()

    model = train_kmeans_model(
        X_train=df,
        preprocessor=preprocessor,
        n_clusters=2,
    )

    preds = model.predict(df)

    # Check prediction length matches input length
    assert len(preds) == len(df)

    # Check cluster labels are integers
    assert all(isinstance(p, (int, np.integer)) for p in preds)


def test_invalid_cluster_number():
    """
    Test that invalid cluster configuration raises error.
    """
    df = build_dummy_dataframe()
    preprocessor = build_dummy_preprocessor()

    with pytest.raises(ValueError):
        train_kmeans_model(
            X_train=df,
            preprocessor=preprocessor,
            n_clusters=0,  # invalid
        )