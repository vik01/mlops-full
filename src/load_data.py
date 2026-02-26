"""
Educational Goal:
- Why this module exists in an MLOps system:
    Isolate raw data ingestion so the pipeline does not depend on notebook
    state.
- Responsibility (separation of concerns): Only load raw data from disk. No
  splitting, no cleaning, no feature engineering.
- Pipeline contract (inputs and outputs): Input is a Path to a CSV file;
  output is a pandas DataFrame.

TODO: Replace print statements with standard library logging in a later
      session
TODO: Any temporary or hardcoded variable or parameter will be imported
      from config.yml in a later session
"""

from pathlib import Path
import pandas as pd

from src.utils import load_csv


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path to a raw CSV file (train.csv or test.csv)
    Outputs:
    - df_raw: DataFrame containing raw data
    Why this contract matters for reliable ML delivery:
    - Centralizing data loading ensures consistent ingestion behavior
      across training, evaluation, and inference environments.
    """
    print(
        f"[load_data.load_raw_data] Loading raw data from: {raw_data_path}"
    )  # TODO: replace with logging later

    # Fail fast if file does not exist
    # This prevents silent training on wrong or missing data.
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"[load_data] File not found at {raw_data_path}. "
            "Check your SETTINGS configuration and ensure the file "
            "exists in data/raw/."
        )

    # Load CSV using centralized utility function
    # This ensures consistent read behavior across the entire pipeline.
    df_raw = load_csv(raw_data_path)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add ingestion-time adjustments ONLY if strictly
    # necessary. Why: Some datasets may require dtype casting or date
    # parsing at load time.
    #
    # IMPORTANT:
    # - Do NOT perform cleaning here.
    # - Do NOT perform feature engineering here.
    # - Do NOT split train/test here.
    #
    # Examples (only if needed):
    # 1. df_raw["date"] = pd.to_datetime(df_raw["date"])
    # 2. df_raw["some_numeric_col"] = (
    #    df_raw["some_numeric_col"].astype("int64"))
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError(
    #     "Student: You must implement this logic to proceed!"
    # )
    #
    # Since Kaggle CSV files are already clean and well formatted,
    # no ingestion adjustments are currently required.
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return df_raw
