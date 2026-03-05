"""
Educational Goal:
- Why this module exists in an MLOps system: Cleaning logic is often the most
dataset-specific part of an ML pipeline.
  By isolating it here, we prevent notebooks from becoming the single source
  of truth and we make cleaning reproducible.
- Responsibility (separation of concerns): This module transforms raw data
into a "model-ready" table (still tabular),
  without doing feature engineering (handled in features.py) or model fitting
  (handled in train.py).
- Pipeline contract (inputs and outputs):
  Inputs : raw pandas DataFrame + target column name
  Outputs: cleaned pandas DataFrame (same target column retained)

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: Raw input DataFrame loaded from disk (may contain messy values,
    missing rows, etc.)
    - target_column: Name of the target label column (string). Included to
    make the cleaning step configurable.
    Outputs:
    - df_clean: Cleaned DataFrame that will be validated, split, and used
    downstream.
    Why this contract matters for reliable ML delivery:
    - Keeping a stable input/output contract lets other pipeline stages
    (validation, features, training)
      remain unchanged even as cleaning logic evolves per dataset or business
      rules.
    """
    print("[clean_data] Cleaning raw dataframe...")
    # TODO: replace with logging later

    if df_raw is None:
        raise ValueError(
            "[clean_data] df_raw is None.load_raw_data() must return a pd DF.")

    # ------------------------------------------------------------------
    # Baseline logic (must run end-to-end out of the box)
    # ------------------------------------------------------------------
    # IMPORTANT: baseline is an identity transform to keep scaffolding
    # runnable.
    # We still accept `target_column` to match the pipeline contract used in
    # main.py.
    df_clean = df_raw.copy(deep=True)

    df_clean = df_clean.dropna(how="all")
    df_clean = df_clean.drop_duplicates()

    # Minimal contract visibility: ensure we didn't accidentally drop the
    # target silently.
    # We do NOT fail here (validate.py is the place for fail-fast required
    # columns),
    # but we print a helpful warning for students.
    if target_column not in df_clean.columns:
        print(
            f"[clean_data] Warning: target_column='{target_column}' not found"
            "Downstream validation will likely fail if this is unintended."
        )  # TODO: replace with logging later

    return df_clean
