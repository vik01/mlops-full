import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from src.dtrees.dtrees_train import train_model
from src.dtrees.dtrees_eval import evaluate_model, calculate_metrics


@pytest.fixture
def eval_setup():
    np.random.seed(42)
    X = pd.DataFrame({
        "age": np.random.randint(30, 80, 300),
        "bp":  np.random.randint(60, 140, 300),
    })
    y = pd.Series(np.random.choice(["Presence", "Absence"], size=300))
    preprocessor = ColumnTransformer([
        ("scaler", StandardScaler(), ["age", "bp"])
    ])
    split = 240
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = train_model(X_train, y_train, preprocessor, "classification")
    return model, X_test, y_test


def test_evaluate_model_returns_float(eval_setup):
    """evaluate_model should return a single float."""
    model, X_test, y_test = eval_setup
    result = evaluate_model(model, X_test, y_test, "classification")
    assert isinstance(result, float)


def test_evaluate_model_f1_between_0_and_1(eval_setup):
    """Returned F1 score should be between 0 and 1."""
    model, X_test, y_test = eval_setup
    result = evaluate_model(model, X_test, y_test, "classification")
    assert 0.0 <= result <= 1.0


def test_calculate_metrics_returns_all_keys():
    """calculate_metrics should return all expected metric keys."""
    y_true = pd.Series(["Presence", "Absence", "Presence", "Absence"])
    y_pred = np.array(["Presence", "Absence", "Absence", "Absence"])
    cm = confusion_matrix(y_true, y_pred, labels=["Absence", "Presence"])
    metrics = calculate_metrics(cm, y_true, y_pred)
    expected_keys = {"TP", "TN", "FP", "FN", "Accuracy", "Precision",
                     "Recall", "Specificity", "F1-score", "False Positive Rate"}
    assert expected_keys == set(metrics.keys())


def test_calculate_metrics_accuracy_range():
    """Accuracy returned by calculate_metrics should be between 0 and 1."""
    y_true = pd.Series(["Presence", "Absence", "Presence", "Absence"])
    y_pred = np.array(["Presence", "Presence", "Presence", "Absence"])
    cm = confusion_matrix(y_true, y_pred, labels=["Absence", "Presence"])
    metrics = calculate_metrics(cm, y_true, y_pred)
    assert 0.0 <= metrics["Accuracy"] <= 1.0


def test_evaluate_model_raises_on_bad_input(eval_setup):
    """evaluate_model should raise RuntimeError with incompatible input."""
    model, _, y_test = eval_setup
    bad_X = pd.DataFrame({"wrong": [1, 2, 3]})
    bad_y = y_test.iloc[:3]
    with pytest.raises(RuntimeError):
        evaluate_model(model, bad_X, bad_y, "classification")