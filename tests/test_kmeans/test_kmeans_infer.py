import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from kmeans.infer import run_kmeans_inference


def build_dummy_preprocessor():
    """
    Minimal preprocessor compatible with KMeans pipeline.
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
    Small deterministic dataset for inference testing.
    """
    return pd.DataFrame(
        {
            "num_feature": [0, 1, 2, 3, 4, 5],
            "cat_feature": ["A", "B", "A", "B", "A", "B"],
        }
    )


def build_fitted_model():
    """
    Create and fit a simple KMeans pipeline.
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


def test_run_kmeans_inference_returns_dataframe():
    """
    Ensure inference returns a DataFrame with correct structure.
    """
    model, df = build_fitted_model()

    result = run_kmeans_inference(model, df)

    # Type check
    assert isinstance(result, pd.DataFrame)

    # Column name check
    assert list(result.columns) == ["prediction"]

    # Length check
    assert len(result) == len(df)

    # Index preservation check
    assert all(result.index == df.index)


def test_prediction_values_are_integer():
    """
    Ensure cluster labels are integer-like.
    """
    model, df = build_fitted_model()

    result = run_kmeans_inference(model, df)

    preds = result["prediction"].values

    # Accept numpy integer types
    assert all(isinstance(p, (int, np.integer)) for p in preds)