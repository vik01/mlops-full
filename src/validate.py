"""
Educational Goal:
- Why this module exists in an MLOps system: Data is the most common failure point in
  ML pipelines. A validation layer catches schema mismatches, empty DataFrames, and
  missing columns early — before they cause cryptic errors inside scikit-learn or,
  worse, silently produce a model trained on the wrong data.
- Responsibility (separation of concerns): validate.py owns ONLY data quality checks.
  It does not transform data or make predictions. It either confirms the data is safe
  to use or raises a clear, actionable error.
- Pipeline contract (inputs and outputs):
    Inputs : a cleaned pd.DataFrame and a list of column names that must be present
    Outputs: True if all checks pass, raises ValueError or TypeError if not

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df              : the cleaned pd.DataFrame to validate
    - required_columns: list of column name strings that must exist in the DataFrame
    Outputs:
    - True if all checks pass
    - Raises ValueError if the DataFrame is empty or a required column is missing
    Why this contract matters for reliable ML delivery:
    - Catching data issues here — before training — means errors are surfaced with
      clear, human-readable messages rather than cryptic scikit-learn or numpy errors.
      It also means the pipeline fails loudly at the right step, making debugging fast.
    """
    print("[validate] Running data validation checks...")  # TODO: replace with logging later

    # ------------------------------------------------------------------
    # CHECK 1: DataFrame must not be empty
    # An empty DataFrame almost always means a broken data path or a
    # filtering step that removed all rows. Fail immediately.
    # ------------------------------------------------------------------
    if df.empty:
        raise ValueError(
            "[validate] The DataFrame is empty. "
            "Check your data path and any filtering steps in clean_data.py."
        )

    # ------------------------------------------------------------------
    # CHECK 2: All required columns must be present
    # The required_columns list is built in main.py from SETTINGS, so
    # this check enforces that the SETTINGS config matches the actual data.
    # ------------------------------------------------------------------
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"[validate] The following required columns are missing from the DataFrame: {missing}\n"
            "Check that your SETTINGS in main.py match the actual column names in your CSV."
        )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add dataset-specific quality checks here.
    # Why: Every dataset has its own rules for what "valid" data looks like.
    #      Generic checks above catch obvious schema issues, but you should
    #      add checks for your specific columns and business logic.
    # Examples:
    # 1. Check for too many nulls in a critical column:
    #    null_rate = df["Age"].isnull().mean()
    #    if null_rate > 0.1:
    #        raise ValueError(f"[validate] 'Age' has {null_rate:.0%} nulls — too many to proceed.")
    #
    # 2. Check that the target column contains only expected values:
    #    valid_targets = {0, 1}
    #    unexpected = set(df["Heart Disease"].unique()) - valid_targets
    #    if unexpected:
    #        raise ValueError(f"[validate] Unexpected target values: {unexpected}")
    #
    # 3. Check that a numeric column has no negative values:
    #    if (df["Age"] < 0).any():
    #        raise ValueError("[validate] 'Age' contains negative values.")
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print(f"[validate]   Shape: {df.shape[0]} rows x {df.shape[1]} columns.")  # TODO: replace with logging later
    print(f"[validate]   All {len(required_columns)} required columns present.")  # TODO: replace with logging later
    print("[validate] Validation passed.")  # TODO: replace with logging later

    return True
