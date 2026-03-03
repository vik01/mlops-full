import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.dtrees.dtrees_train import train_model
from src.dtrees.dtrees_infer import run_inference


@pytest.fixture
def trained_pipeline():
    np.random.seed(42)
    X = pd.DataFrame({
        "age": np.random.randint(30, 80, 200),
        "bp":  np.random.randint(60, 140, 200),
    })
    y = pd.Series(np.random.choice(["Presence", "Absence"], size=200))
    preprocessor = ColumnTransformer([
        ("scaler", StandardScaler(), ["age", "bp"])
    ])
    return train_model(X, y, preprocessor, "classification")


@pytest.fixture
def infer_data():
    np.random.seed(0)
    return pd.DataFrame({
        "age": np.random.randint(30, 80, 10),
        "bp":  np.random.randint(60, 140, 10),
    }, index=range(100, 110))


def test_run_inference_returns_dataframe(trained_pipeline, infer_data):
    """run_inference should return a pandas DataFrame."""
    result = run_inference(trained_pipeline, infer_data)
    assert isinstance(result, pd.DataFrame)


def test_output_has_prediction_column(trained_pipeline, infer_data):
    """Output DataFrame should have a 'prediction' column."""
    result = run_inference(trained_pipeline, infer_data)
    assert "prediction" in result.columns


def test_output_preserves_input_index(trained_pipeline, infer_data):
    """Output index should match the input index exactly."""
    result = run_inference(trained_pipeline, infer_data)
    assert list(result.index) == list(infer_data.index)


def test_predictions_are_valid_classes(trained_pipeline, infer_data):
    """All predictions should be either 'Presence' or 'Absence'."""
    result = run_inference(trained_pipeline, infer_data)
    assert set(result["prediction"]).issubset({"Presence", "Absence"})


def test_output_length_matches_input(trained_pipeline, infer_data):
    """Number of predictions should match number of input rows."""
    result = run_inference(trained_pipeline, infer_data)
    assert len(result) == len(infer_data)


def test_run_inference_raises_on_bad_input(trained_pipeline):
    """run_inference should raise RuntimeError on incompatible input."""
    bad_input = pd.DataFrame({"wrong_col": [1, 2, 3]})
    with pytest.raises(RuntimeError):
        run_inference(trained_pipeline, bad_input)