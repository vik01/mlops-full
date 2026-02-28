"""
Tests for src/load_data.py — load_raw_data
Run with: pytest tests/test_load_data.py
"""

import sys
from pathlib import Path

import pandas as pd

# Add the project root so that "from src.utils import ..." inside load_data.py resolves correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load_data import load_raw_data


MOCK_DATA_PATH = Path(__file__).parent / "mock_data" / "test_train.csv"


# ------------------------------------------------------------------
# load_raw_data: returned value is a DataFrame and column types
# match the expectations defined in load_data.py
# ------------------------------------------------------------------
def test_load_raw_data_types():
    df = load_raw_data(MOCK_DATA_PATH)

    # --- 1. Return type ---
    assert isinstance(df, pd.DataFrame)

    # --- 2. int64 columns (taken directly from int_64_cols in load_data.py) ---
    int64_cols = [
        "id",
        "Age",
        "Sex",
        "Chest pain type",
        "BP",
        "Cholesterol",
        "FBS over 120",
        "EKG results",
        "Max HR",
        "Exercise angina",
        "Slope of ST",
        "Number of vessels fluro",
        "Thallium",
    ]
    for col in int64_cols:
        assert df[col].dtype == "int64", f"Expected int64 for '{col}', got {df[col].dtype}"

    # --- 3. float64 column (taken directly from float_64_cols in load_data.py) ---
    assert df["ST depression"].dtype == "float64", (
        f"Expected float64 for 'ST depression', got {df['ST depression'].dtype}"
    )

    # --- 4. string column (taken directly from str_cols in load_data.py) ---
    # pandas stores str columns as object dtype
    assert df["Heart Disease"].dtype == "str", (
        f"Expected object (str) for 'Heart Disease', got {df['Heart Disease'].dtype}"
    )
