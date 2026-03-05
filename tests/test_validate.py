"""
Tests for src/validate.py
Run with: pytest tests/test_validate.py
"""

from pathlib import Path
import pandas as pd
import pytest
from validate import validate_dataframe


MOCK_DATA_PATH = Path(__file__).parent / "mock_data" / "test_train.csv"

REQUIRED_COLUMNS = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
    "Heart Disease",
]


@pytest.fixture(scope="module")
def mock_df():
    """Load mock CSV once per module and drop the id column
    (simulates post-cleaning state)."""
    df = pd.read_csv(MOCK_DATA_PATH)
    df = df.drop(columns=["id"])
    return df


# ------------------------------------------------------------------
# CHECK 1: DataFrame must not be empty
# ------------------------------------------------------------------
def test_check1_empty_dataframe(mock_df):
    df = mock_df.copy()
    df = df.iloc[0:0]  # keep columns but remove all rows
    with pytest.raises(ValueError):
        validate_dataframe(df, REQUIRED_COLUMNS)


# ------------------------------------------------------------------
# CHECK 2: All required columns must be present
# ------------------------------------------------------------------
def test_check2_missing_required_columns(mock_df):
    df = mock_df.copy()
    df = df.drop(columns=["Age"])
    with pytest.raises(ValueError):
        validate_dataframe(df, REQUIRED_COLUMNS)


# ------------------------------------------------------------------
# CHECK 3: No column may have more than 10% null values
# ------------------------------------------------------------------
def test_check3_high_null_rate(mock_df):
    df = mock_df.copy()
    # Set 4 of 28 rows (~14%) in Cholesterol to NaN — exceeds the 10% threshold
    df.loc[0, "Cholesterol"] = None
    df.loc[1, "Cholesterol"] = None
    df.loc[2, "Cholesterol"] = None
    df.loc[3, "Cholesterol"] = None
    with pytest.raises(ValueError):
        validate_dataframe(df, REQUIRED_COLUMNS)


# ------------------------------------------------------------------
# CHECK 4: Target column must contain only "Presence" or "Absence"
# ------------------------------------------------------------------
def test_check4_invalid_target_values(mock_df):
    df = mock_df.copy()
    df.loc[0, "Heart Disease"] = "Unknown"
    with pytest.raises(ValueError):
        validate_dataframe(df, REQUIRED_COLUMNS)


# ------------------------------------------------------------------
# CHECK 5: Numeric columns must not contain negative values
# ------------------------------------------------------------------
def test_check5_negative_numeric_values(mock_df):
    df = mock_df.copy()
    df.loc[0, "BP"] = -1
    with pytest.raises(ValueError):
        validate_dataframe(df, REQUIRED_COLUMNS)


# ------------------------------------------------------------------
# CHECK 6: No duplicate rows
# ------------------------------------------------------------------
def test_check6_duplicate_rows(mock_df):
    df = mock_df.copy()
    duplicate_row = df.iloc[[0]]
    df = pd.concat([df, duplicate_row], ignore_index=True)
    with pytest.raises(ValueError):
        validate_dataframe(df, REQUIRED_COLUMNS)


# ------------------------------------------------------------------
# CHECK 7: Binary columns must contain only 0 or 1
# ------------------------------------------------------------------
def test_check7_invalid_binary_values(mock_df):
    df = mock_df.copy()
    df.loc[0, "Sex"] = 2
    with pytest.raises(ValueError):
        validate_dataframe(df, REQUIRED_COLUMNS)


# ------------------------------------------------------------------
# CHECK 8: Age must be within medically plausible range (0–120)
# ------------------------------------------------------------------
def test_check8_age_out_of_range(mock_df):
    df = mock_df.copy()
    df.loc[0, "Age"] = 200
    with pytest.raises(ValueError):
        validate_dataframe(df, REQUIRED_COLUMNS)


# ------------------------------------------------------------------
# ALL CHECKS PASS: Clean mock data should return True
# ------------------------------------------------------------------
def test_all_checks_pass(mock_df):
    df = mock_df.copy()
    result = validate_dataframe(df, REQUIRED_COLUMNS)
    assert result is True
