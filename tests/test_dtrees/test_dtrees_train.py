import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification


@pytest.fixture
def sample_training_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    return X, y


def test_model_trains_without_error(sample_training_data):
    """Model should fit on valid training data without raising errors."""
    X, y = sample_training_data
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    assert model is not None


def test_model_has_correct_classes(sample_training_data):
    """Trained model should have the correct number of unique classes."""
    X, y = sample_training_data
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    assert len(model.classes_) == len(np.unique(y))


def test_model_tree_depth_is_bounded(sample_training_data):
    """Model trained with max_depth should not exceed that depth."""
    X, y = sample_training_data
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    assert model.get_depth() <= 5


def test_model_fails_on_mismatched_shapes():
    """Model should raise an error if X and y have mismatched lengths."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, size=80)  # wrong size
    model = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_model_feature_importances_sum_to_one(sample_training_data):
    """Feature importances should sum to 1 after training."""
    X, y = sample_training_data
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    assert pytest.approx(sum(model.feature_importances_), 0.001) == 1.0
    