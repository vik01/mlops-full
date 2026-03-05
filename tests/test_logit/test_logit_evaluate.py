import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from logit_regression.logit_train import train_logit_model
from logit_regression.logit_evaluate import evaluate_logit_model


def _make_preprocessor():
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("cat", ohe, ["cat_feature"]),
            ("num", "passthrough", ["num_feature"]),
        ],
        remainder="drop",
    )


def _make_dummy_classification_data():
    df = pd.DataFrame(
        {
            "num_feature": [0.0, 1.0, 2.0, 3.0],
            "cat_feature": ["A", "B", "A", "C"],
            "target": [0, 1, 0, 1],
        }
    )
    X = df[["num_feature", "cat_feature"]]
    y = df["target"]
    return X, y


def test_evaluate_model_returns_float_classification():
    X, y = _make_dummy_classification_data()
    preprocessor = _make_preprocessor()

    model = train_logit_model(
        X_train=X,
        y_train=y,
        preprocessor=preprocessor,
        problem_type="classification",
    )

    metric_value = evaluate_logit_model(
        model=model,
        X_test=X,
        y_test=y,
        problem_type="classification",
    )

    assert isinstance(metric_value, float)
