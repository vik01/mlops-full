"""
Tests for src/features.py — get_feature_preprocessor
Run with: pytest tests/test_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import get_feature_preprocessor


# ===========================================================================
# INDIVIDUAL FEATURE TESTS
# Each test passes only one populated list and asserts:
#   - the transformers list has exactly 1 entry
#   - the name at index 0 of that tuple is correct
# ===========================================================================

# ------------------------------------------------------------------
# quantile_bin: single column → one transformer named "quantile_bin"
# ------------------------------------------------------------------
def test_single_quantile_bin():
    preprocessor = get_feature_preprocessor(quantile_bin_cols=["col"])
    transformers = preprocessor.transformers # type: ignore
    assert len(transformers) == 1
    assert transformers[0][0] == "quantile_bin"


# ------------------------------------------------------------------
# onehot: single column → one transformer named "onehot"
# ------------------------------------------------------------------
def test_single_onehot():
    preprocessor = get_feature_preprocessor(categorical_onehot_cols=["col"])
    transformers = preprocessor.transformers # type: ignore
    assert len(transformers) == 1
    assert transformers[0][0] == "onehot"


# ------------------------------------------------------------------
# passthrough: single column → one transformer named "passthrough"
# ------------------------------------------------------------------
def test_single_passthrough():
    preprocessor = get_feature_preprocessor(numeric_passthrough_cols=["col"])
    transformers = preprocessor.transformers # type: ignore
    assert len(transformers) == 1
    assert transformers[0][0] == "passthrough"


# ------------------------------------------------------------------
# scaler: single column → one transformer named "scaler"
# ------------------------------------------------------------------
def test_single_scaler():
    preprocessor = get_feature_preprocessor(min_max_cols=["col"])
    transformers = preprocessor.transformers # type: ignore
    assert len(transformers) == 1
    assert transformers[0][0] == "scaler"


# ------------------------------------------------------------------
# ordinal: single column → one transformer named "ordinal"
# ------------------------------------------------------------------
def test_single_ordinal():
    preprocessor = get_feature_preprocessor(ordinal_encode_cols=["col"])
    transformers = preprocessor.transformers # type: ignore
    assert len(transformers) == 1
    assert transformers[0][0] == "ordinal"


# ===========================================================================
# SETTINGS TEST
# Uses the exact column lists from SETTINGS["features"] in main.py.
# categorical_onehot is empty so "onehot" is skipped, leaving 4 transformers
# in this order: quantile_bin, passthrough, scaler, ordinal
# ===========================================================================
def test_settings_feature_config():
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["ST depression"],
        categorical_onehot_cols=[],
        numeric_passthrough_cols=["id", "Sex", "FBS over 120", "EKG results", "Exercise angina"],
        min_max_cols=["Age", "BP", "Cholesterol", "Max HR"],
        ordinal_encode_cols=["Chest pain type", "Number of vessels fluro", "Thallium", "Slope of ST"],
        n_bins=5,
    )
    transformers = preprocessor.transformers # type: ignore
    assert len(transformers) == 4
    assert transformers[0][0] == "quantile_bin"
    assert transformers[1][0] == "passthrough"
    assert transformers[2][0] == "scaler"
    assert transformers[3][0] == "ordinal"
