import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from dtrees.dtrees_train import train_dtrees_model


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "age": np.random.randint(30, 80, 200),
        "bp":  np.random.randint(60, 140, 200),
    })
    y = pd.Series(
        np.random.choice(["Presence", "Absence"], size=200),
        name="target"
    )
    return X, y


@pytest.fixture
def preprocessor():
    return ColumnTransformer([
        ("scaler", StandardScaler(), ["age", "bp"])
    ])


def test_train_returns_pipeline(sample_data, preprocessor):
    """train_model should return a fitted sklearn Pipeline."""
    X, y = sample_data
    pipeline = train_dtrees_model(X, y, preprocessor, "classification")
    assert isinstance(pipeline, Pipeline)


def test_pipeline_has_correct_steps(sample_data, preprocessor):
    """Pipeline should have a preprocess step and a model step."""
    X, y = sample_data
    pipeline = train_dtrees_model(X, y, preprocessor, "classification")
    assert "preprocess" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_estimator_is_decision_tree(sample_data, preprocessor):
    """The model step should be a DecisionTreeClassifier."""
    X, y = sample_data
    pipeline = train_dtrees_model(X, y, preprocessor, "classification")
    assert isinstance(pipeline.named_steps["model"], DecisionTreeClassifier)


def test_trained_model_respects_max_depth(sample_data, preprocessor):
    """Trained tree depth should not exceed max_depth=4."""
    X, y = sample_data
    pipeline = train_dtrees_model(X, y, preprocessor, "classification")
    assert pipeline.named_steps["model"].get_depth() <= 4


def test_pipeline_raises_on_bad_data(preprocessor):
    """train_model should raise RuntimeError if fitting fails."""
    X_bad = pd.DataFrame({"age": ["not", "a", "number"],
                          "bp": ["x", "y", "z"]})
    y_bad = pd.Series(["Presence", "Absence", "Presence"])
    with pytest.raises(RuntimeError):
        train_dtrees_model(X_bad, y_bad, preprocessor, "classification")


# ---- Input validation tests ----

def test_X_train_not_dataframe(sample_data, preprocessor):
    """Passing a non-DataFrame X_train should raise TypeError."""
    _, y = sample_data
    with pytest.raises(TypeError, match="X_train must be a pandas DataFrame"):
        train_dtrees_model([[1, 2], [3, 4]], y, preprocessor, "classification")


def test_X_train_empty(sample_data, preprocessor):
    """Passing an empty DataFrame should raise ValueError."""
    _, y = sample_data
    X_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="X_train must not be empty"):
        train_dtrees_model(X_empty, y, preprocessor, "classification")


def test_y_train_not_series(sample_data, preprocessor):
    """Passing a non-Series y_train should raise TypeError."""
    X, _ = sample_data
    with pytest.raises(TypeError, match="y_train must be a pandas Series"):
        train_dtrees_model(X, [0, 1, 0], preprocessor, "classification")


def test_y_train_empty(sample_data, preprocessor):
    """Passing an empty Series should raise ValueError."""
    X, _ = sample_data
    y_empty = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="y_train must not be empty"):
        train_dtrees_model(X, y_empty, preprocessor, "classification")


def test_preprocessor_not_column_transformer(sample_data):
    """Passing a non-ColumnTransformer preprocessor should raise TypeError."""
    X, y = sample_data
    with pytest.raises(TypeError, match="preprocessor must be a ColumnTransformer"):
        train_dtrees_model(X, y, "not_a_transformer", "classification")


def test_problem_type_not_string(sample_data, preprocessor):
    """Passing a non-string problem_type should raise TypeError."""
    X, y = sample_data
    with pytest.raises(TypeError, match="problem_type must be a string"):
        train_dtrees_model(X, y, preprocessor, 123)
