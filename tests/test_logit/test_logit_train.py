import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from logit_regression.logit_train import train_logit_model


def _make_preprocessor():
    # OneHotEncoder API differs across sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, ["cat_feature"]),
            ("num", "passthrough", ["num_feature"]),
        ],
        remainder="drop",
    )
    return preprocessor


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


def test_train_model_returns_fitted_pipeline_classification():
    X_train, y_train = _make_dummy_classification_data()
    preprocessor = _make_preprocessor()

    model = train_logit_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type="classification",
    )

    # If it's fitted, it should be able to predict without crashing
    preds = model.predict(X_train)

    assert len(preds) == len(X_train)
