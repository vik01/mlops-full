"""
Tests for src/utils.py — load_csv and save_csv
Run with: pytest tests/test_utils.py
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_csv, save_csv


MOCK_DATA_PATH = Path(__file__).parent / "mock_data" / "test_train.csv"
BAD_PATH = Path(__file__).parent / "mock_data" / "bad_path.csv"


# ------------------------------------------------------------------
# load_csv: a non-existent path raises FileNotFoundError
# ------------------------------------------------------------------
def test_load_csv_bad_path():
    with pytest.raises(FileNotFoundError):
        load_csv(BAD_PATH)


# ------------------------------------------------------------------
# load_csv: a valid path returns a pandas DataFrame
# ------------------------------------------------------------------
def test_load_csv_good_path():
    df = load_csv(MOCK_DATA_PATH)
    assert isinstance(df, pd.DataFrame)


# ------------------------------------------------------------------
# save_csv: parent directory is created when it does not exist
# tmp_path is a built-in pytest fixture that provides a clean
# temporary directory unique to each test run
# ------------------------------------------------------------------
def test_save_csv_creates_parent_dir(tmp_path):
    df = load_csv(MOCK_DATA_PATH)
    nested_path = tmp_path / "new_subdir" / "output.csv"
    save_csv(df, nested_path)
    assert nested_path.parent.exists()
