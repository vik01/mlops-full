"""
TODO: Any temporary or hardcoded variable or parameter will be imported
      from config.yml in a later session
"""

# Standard Library Imports
import logging
from pathlib import Path

# Third-party Imports
import pandas as pd

# Local Module Imports
from utils import load_csv

logger = logging.getLogger(__name__)


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """Will use src.utils.load_csv() to load raw data from a CSV file
    and return a DataFrame.
    Inputs:
    - raw_data_path: Path to a raw CSV file (train.csv or test.csv)
    Outputs:
    - df_raw: DataFrame containing raw data
    """
    logger.info("Loading raw data from: %s", raw_data_path)

    # NOTE: src.utils.load_csv() will raise a
    #       FileNotFoundError if the file doesn't exist.
    df_raw = load_csv(raw_data_path)

    # ------------------------------------------------------------------
    # Make sure column types match expectations
    # ------------------------------------------------------------------
    int_64_cols = ["id", "Age", "Sex", "Chest pain type", "BP",
                   "Cholesterol", "FBS over 120", "EKG results",
                   "Max HR", "Exercise angina", "Slope of ST",
                   "Number of vessels fluro", "Thallium"
                   ]
    str_cols, float_64_cols = ["Heart Disease"], ["ST depression"]
    for columns in df_raw.columns:
        if columns in int_64_cols:
            df_raw[columns] = df_raw[columns].astype("int64")
        elif columns in str_cols:
            df_raw[columns] = df_raw[columns].astype("str")
        elif columns in float_64_cols:
            df_raw[columns] = df_raw[columns].astype("float64")

    # NOTE: Since Kaggle CSV files are already clean and well formatted,
    #       no additional ingestion adjustments are currently required.

    return df_raw
