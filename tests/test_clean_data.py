"""
Tests for src/clean_data.py
Run with: pytest tests/test_clean_data.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clean_data import clean_dataframe


def test_raises_if_df_raw_is_none():
    with pytest.raises(ValueError):
        clean_dataframe(None, target_column="Heart Disease")"""
Tests for src/clean_data.py
Run with: pytest tests/test_clean_data.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clean_data import clean_dataframe


def test_raises_if_df_raw_is_none():
    with pytest.raises(ValueError):
        clean_dataframe(None, target_column="Heart Disease")


def test_drops_all_null_rows():
    df_raw = pd.DataFrame(
        {
            "A": [1, None, 3],
            "B": [4, None, 6],
            "Heart Disease": ["Presence", None, "Absence"],
        }
    )
    df_raw.loc[1, ["A", "B", "Heart Disease"]] = [None, None, None]

    cleaned = clean_dataframe(df_raw, target_column="Heart Disease")

    assert cleaned.shape[0] == 2
    assert not cleaned.isna().all(axis=1).any()


def test_drops_duplicate_rows():
    df_raw = pd.DataFrame(
        {
            "A": [1, 1, 2],
            "B": [10, 10, 20],
            "Heart Disease": ["Presence", "Presence", "Absence"],
        }
    )

    cleaned = clean_dataframe(df_raw, target_column="Heart Disease")

    assert cleaned.duplicated().sum() == 0
    assert cleaned.shape[0] == 2


def test_target_column_is_preserved_when_present():
    df_raw = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [10, 20],
            "Heart Disease": ["Presence", "Absence"],
        }
    )

    cleaned = clean_dataframe(df_raw, target_column="Heart Disease")

    assert "Heart Disease" in cleaned.columns
    assert cleaned["Heart Disease"].tolist() == ["Presence", "Absence"]


def test_prints_warning_when_target_missing(capsys):
    df_raw = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    cleaned = clean_dataframe(df_raw, target_column="Heart Disease")
    captured = capsys.readouterr()

    assert "Warning: target_column='Heart Disease' not found" in captured.out
    assert cleaned.shape[0] == 2


def test_input_dataframe_is_not_modified():
    df_raw = pd.DataFrame(
        {
            "A": [1, 1, None],
            "B": [10, 10, None],
            "Heart Disease": ["Presence", "Presence", None],
        }
    )
    df_raw.loc[2, ["A", "B", "Heart Disease"]] = [None, None, None]

    _ = clean_dataframe(df_raw, target_column="Heart Disease")

    assert df_raw.shape[0] == 3
    assert df_raw.duplicated().sum() == 1
    assert df_raw.isna().all(axis=1).sum() == 1


def test_drops_all_null_rows():
    df_raw = pd.DataFrame(
        {
            "A": [1, None, 3],
            "B": [4, None, 6],
            "Heart Disease": ["Presence", None, "Absence"],
        }
    )
    df_raw.loc[1, ["A", "B", "Heart Disease"]] = [None, None, None]

    cleaned = clean_dataframe(df_raw, target_column="Heart Disease")

    assert cleaned.shape[0] == 2
    assert not cleaned.isna().all(axis=1).any()


def test_drops_duplicate_rows():
    df_raw = pd.DataFrame(
        {
            "A": [1, 1, 2],
            "B": [10, 10, 20],
            "Heart Disease": ["Presence", "Presence", "Absence"],
        }
    )

    cleaned = clean_dataframe(df_raw, target_column="Heart Disease")

    assert cleaned.duplicated().sum() == 0
    assert cleaned.shape[0] == 2


def test_target_column_is_preserved_when_present():
    df_raw = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [10, 20],
            "Heart Disease": ["Presence", "Absence"],
        }
    )

    cleaned = clean_dataframe(df_raw, target_column="Heart Disease")

    assert "Heart Disease" in cleaned.columns
    assert cleaned["Heart Disease"].tolist() == ["Presence", "Absence"]


def test_prints_warning_when_target_missing(capsys):
    df_raw = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    cleaned = clean_dataframe(df_raw, target_column="Heart Disease")
    captured = capsys.readouterr()

    assert "Warning: target_column='Heart Disease' not found" in captured.out
    assert cleaned.shape[0] == 2


def test_input_dataframe_is_not_modified():
    df_raw = pd.DataFrame(
        {
            "A": [1, 1, None],
            "B": [10, 10, None],
            "Heart Disease": ["Presence", "Presence", None],
        }
    )
    df_raw.loc[2, ["A", "B", "Heart Disease"]] = [None, None, None]

    _ = clean_dataframe(df_raw, target_column="Heart Disease")

    assert df_raw.shape[0] == 3
    assert df_raw.duplicated().sum() == 1
    assert df_raw.isna().all(axis=1).sum() == 1