import pytest
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
        random_state=42,
        max_iterations=500,
    )

    # If it's fitted, it should be able to predict without crashing
    preds = model.predict(X_train)

    assert len(preds) == len(X_train)


# ---- Input validation tests ----

def test_X_train_not_dataframe():
    """Passing a non-DataFrame X_train should raise TypeError."""
    _, y = _make_dummy_classification_data()
    preprocessor = _make_preprocessor()
    with pytest.raises(TypeError, match="X_train must be a pandas DataFrame"):
        train_logit_model([[1, 2]], y, preprocessor, "classification", 42, 500)


def test_X_train_empty():
    """Passing an empty DataFrame should raise ValueError."""
    _, y = _make_dummy_classification_data()
    preprocessor = _make_preprocessor()
    with pytest.raises(ValueError, match="X_train must not be empty"):
        train_logit_model(pd.DataFrame(), y, preprocessor, "classification", 42, 500)


def test_y_train_not_series():
    """Passing a non-Series y_train should raise TypeError."""
    X, _ = _make_dummy_classification_data()
    preprocessor = _make_preprocessor()
    with pytest.raises(TypeError, match="y_train must be a pandas Series"):
        train_logit_model(X, [0, 1, 0], preprocessor, "classification", 42, 500)


def test_y_train_empty():
    """Passing an empty Series should raise ValueError."""
    X, _ = _make_dummy_classification_data()
    preprocessor = _make_preprocessor()
    y_empty = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="y_train must not be empty"):
        train_logit_model(X, y_empty, preprocessor, "classification", 42, 500)


def test_preprocessor_not_column_transformer():
    """Passing a non-ColumnTransformer preprocessor should raise TypeError."""
    X, y = _make_dummy_classification_data()
    with pytest.raises(TypeError, match="preprocessor must be a ColumnTransformer"):
        train_logit_model(X, y, "not_a_transformer", "classification", 42, 500)


def test_problem_type_not_string():
    """Passing a non-string problem_type should raise TypeError."""
    X, y = _make_dummy_classification_data()
    preprocessor = _make_preprocessor()
    with pytest.raises(TypeError, match="problem_type must be a string"):
        train_logit_model(X, y, preprocessor, 123, 42, 500)


def test_random_state_not_int():
    """Passing a non-int random_state should raise TypeError."""
    X, y = _make_dummy_classification_data()
    preprocessor = _make_preprocessor()
    with pytest.raises(TypeError, match="random_state must be an int"):
        train_logit_model(X, y, preprocessor, "classification", "abc", 500)


def test_max_iterations_not_int():
    """Passing a non-int max_iterations should raise TypeError."""
    X, y = _make_dummy_classification_data()
    preprocessor = _make_preprocessor()
    with pytest.raises(TypeError, match="max_iterations must be an int"):
        train_logit_model(X, y, preprocessor, "classification", 42, 5.5)
