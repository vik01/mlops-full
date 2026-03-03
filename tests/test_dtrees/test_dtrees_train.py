import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.dtrees.dtrees_train import train_model


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
    pipeline = train_model(X, y, preprocessor, "classification")
    assert isinstance(pipeline, Pipeline)


def test_pipeline_has_correct_steps(sample_data, preprocessor):
    """Pipeline should have a preprocess step and a model step."""
    X, y = sample_data
    pipeline = train_model(X, y, preprocessor, "classification")
    assert "preprocess" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_estimator_is_decision_tree(sample_data, preprocessor):
    """The model step should be a DecisionTreeClassifier."""
    X, y = sample_data
    pipeline = train_model(X, y, preprocessor, "classification")
    assert isinstance(pipeline.named_steps["model"], DecisionTreeClassifier)


def test_trained_model_respects_max_depth(sample_data, preprocessor):
    """Trained tree depth should not exceed max_depth=4."""
    X, y = sample_data
    pipeline = train_model(X, y, preprocessor, "classification")
    assert pipeline.named_steps["model"].get_depth() <= 4


def test_pipeline_raises_on_bad_data(preprocessor):
    """train_model should raise RuntimeError if fitting fails."""
    X_bad = pd.DataFrame({"age": ["not", "a", "number"], "bp": ["x", "y", "z"]})
    y_bad = pd.Series(["Presence", "Absence", "Presence"])
    with pytest.raises(RuntimeError):
        train_model(X_bad, y_bad, preprocessor, "classification")