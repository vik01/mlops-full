import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification


@pytest.fixture
def trained_model():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_input():
    return np.random.rand(5, 10)  # 5 samples, 10 features


def test_predict_returns_correct_shape(trained_model, sample_input):
    """Predictions should have one output per input sample."""
    preds = trained_model.predict(sample_input)
    assert preds.shape == (sample_input.shape[0],)


def test_predict_returns_valid_classes(trained_model, sample_input):
    """All predictions should be one of the known class labels."""
    preds = trained_model.predict(sample_input)
    assert all(p in trained_model.classes_ for p in preds)


def test_predict_proba_returns_probabilities(trained_model, sample_input):
    """Probability outputs should be between 0 and 1 and sum to ~1 per row."""
    proba = trained_model.predict_proba(sample_input)
    assert proba.shape == (sample_input.shape[0], len(trained_model.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_single_sample(trained_model):
    """Model should handle a single sample as input."""
    single = np.random.rand(1, 10)
    pred = trained_model.predict(single)
    assert len(pred) == 1


def test_predict_fails_on_wrong_feature_count(trained_model):
    """Model should raise an error if input has wrong number of features."""
    bad_input = np.random.rand(5, 7)  # model trained on 10 features
    with pytest.raises(ValueError):
        trained_model.predict(bad_input)
        