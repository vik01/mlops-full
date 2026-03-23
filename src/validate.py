"""
TODO: Any temporary or hardcoded variable or parameter will be
     imported from config.yml in a later session
"""

# Standard Library Imports
import logging
from pathlib import Path

# Third-party Imports
import pandas as pd

# Local Module Imports 
from utils import load_config, require_section, require_str 
from utils import require_float, require_int, require_list

logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parents[1]

# -----------------------------
# Load and validate config.yaml
# -----------------------------
cfg = load_config(project_root / "config.yaml")

# Load required sections from config file
paths_cfg = require_section(cfg, "paths")
validation_cfg = require_section(cfg, "validation")
problem_cfg = require_section(cfg, "problem")

# Get configurations
TARGET_COLUMN = require_str(problem_cfg, "target_column")
VALID_TARGETS = require_list(problem_cfg, "valid_targets")
MAX_NULL_RATE = require_float(validation_cfg, "max_null_rate")
BINARY_COLUMNS = require_list(validation_cfg, "binary_columns")
AGE_MIN = require_int(validation_cfg, "age_min")
AGE_MAX = require_int(validation_cfg, "age_max")


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df              : the cleaned pd.DataFrame to validate
    - required_columns: list of column names that must exist in DataFrame
    Outputs:
    - True if all checks pass
    - Raises ValueError if the DataFrame misses required columns, has too many
      nulls, contains unexpected target values, or fails any other checks.
    """
    logger.info("Running data validation checks...")

    # ------------------------------------------------------------------
    # CHECK 1: DataFrame must not be empty
    # ------------------------------------------------------------------
    if df.empty:
        raise ValueError(
            "[validate] The DataFrame is empty. "
            "Check your data path and any filtering steps in "
            "clean_data.py."
        )

    # ------------------------------------------------------------------
    # CHECK 2: All required columns must be present
    # ------------------------------------------------------------------
    missing = [col for col in required_columns
               if col not in df.columns]
    if missing:
        raise ValueError(
            f"[validate] The following required columns are missing "
            f"from the DataFrame: {missing}\n"
            "Check that your SETTINGS in main.py match the actual "
            "column names in your CSV."
        )

    # ------------------------------------------------------------------
    # CHECK 3: Check for nulls (data must have less than 10% nulls in
    # any column)
    # ------------------------------------------------------------------
    null_rates = df.isnull().mean()
    high_null_cols = null_rates[null_rates > MAX_NULL_RATE].index.tolist()
    if high_null_cols:
        raise ValueError(
            f"[validate] The following columns have more than 10% "
            f"nulls: {high_null_cols}\n"
            "Consider imputing or dropping these columns."
        )

    # ------------------------------------------------------------------
    # CHECK 4: Check that the target column contains only expected
    # values
    # ------------------------------------------------------------------
    unexpected = (set(df[TARGET_COLUMN].unique()) -
                  set(VALID_TARGETS))
    if unexpected:
        raise ValueError(
            f"[validate] Unexpected target values in '{TARGET_COLUMN}' "
            f"column: {unexpected}\n"
            f"Expected values are: {VALID_TARGETS}"
        )

    # ------------------------------------------------------------------
    # CHECK 5: Check that numeric columns have no negative values
    # (example of a custom check)
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            raise ValueError(
                f"[validate] Numeric column '{col}' contains "
                f"negative values."
            )

    # ------------------------------------------------------------------
    # CHECK 6: No duplicate rows
    # ------------------------------------------------------------------
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        raise ValueError(
            f"[validate] DataFrame contains {n_duplicates} duplicate "
            f"row(s).\n"
            "Remove duplicates before training to avoid data leakage."
        )

    # ------------------------------------------------------------------
    # CHECK 7: Binary columns must contain only 0 or 1
    # 'Sex', 'FBS over 120', and 'Exercise angina' are binary flags.
    # ------------------------------------------------------------------
    binary_cols = BINARY_COLUMNS
    valid_binary = {0, 1}
    for col in binary_cols:
        if col in df.columns:
            unexpected_vals = set(df[col].unique()) - valid_binary
            if unexpected_vals:
                raise ValueError(
                    f"[validate] Binary column '{col}' contains "
                    f"unexpected values: {unexpected_vals}\n"
                    f"Expected only {{0, 1}}. Check your encoding "
                    f"step in clean_data.py."
                )

    # ------------------------------------------------------------------
    # CHECK 8: Age must be within a medically plausible range (0–120)
    # ------------------------------------------------------------------
    if "Age" in df.columns:
        out_of_range = df[(df["Age"] < AGE_MIN) |
                          (df["Age"] > AGE_MAX)]
        if not out_of_range.empty:
            raise ValueError(
                f"[validate] 'Age' column contains {len(out_of_range)} "
                f"value(s) outside the expected range "
                f"[{AGE_MIN}, {AGE_MAX}].\n"
                "Check your raw data source for data entry errors."
            )

    rows, cols = df.shape
    logger.info("Shape: %d rows x %d columns.", rows, cols)
    logger.info("All %d required columns present.", len(required_columns))
    logger.info("Validation passed.")

    return True
