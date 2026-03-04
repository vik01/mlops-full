import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from logit_regression.logit_train import train_logit_model
from logit_regression.logit_infer import run_logit_inference


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


def test_run_inference_output_format():
    X, y = _make_dummy_classification_data()
    preprocessor = _make_preprocessor()

    model = train_logit_model(
        X_train=X,
        y_train=y,
        preprocessor=preprocessor,
        problem_type="classification",
    )

    X_infer = X.head(2).copy()
    preds_df = run_logit_inference(model=model, X_infer=X_infer)

    assert isinstance(preds_df, pd.DataFrame)
    assert list(preds_df.columns) == ["prediction"]
    # must be exactly one column with this name
    assert preds_df.index.equals(X_infer.index)      # must preserve index
    assert len(preds_df) == len(X_infer)