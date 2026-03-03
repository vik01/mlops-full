import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


@pytest.fixture
def eval_data():
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred


def test_accuracy_above_threshold(eval_data):
    """Model accuracy should be above a minimum acceptable threshold."""
    y_test, y_pred = eval_data
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.75, f"Accuracy {acc:.2f} is below the 0.75 threshold"


def test_f1_score_above_threshold(eval_data):
    """F1 score should be above a minimum acceptable threshold."""
    y_test, y_pred = eval_data
    f1 = f1_score(y_test, y_pred, average="weighted")
    assert f1 >= 0.75, f"F1 score {f1:.2f} is below the 0.75 threshold"


def test_confusion_matrix_shape(eval_data):
    """Confusion matrix should be square with size equal to number of classes."""
    y_test, y_pred = eval_data
    cm = confusion_matrix(y_test, y_pred)
    n_classes = len(np.unique(y_test))
    assert cm.shape == (n_classes, n_classes)


def test_no_nan_in_predictions(eval_data):
    """Predictions should not contain NaN values."""
    _, y_pred = eval_data
    assert not np.any(np.isnan(y_pred))


def test_predictions_match_expected_length(eval_data):
    """Number of predictions should match number of test samples."""
    y_test, y_pred = eval_data
    assert len(y_pred) == len(y_test)