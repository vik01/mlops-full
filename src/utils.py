"""
Educational Goal:
- Why this module exists in an MLOps system: Every pipeline needs to read and write
  files — raw data, processed data, trained models, and predictions. Centralising all
  I/O operations in one place means the rest of the codebase never needs to know WHERE
  or HOW data is stored. If the storage format changes (e.g. CSV to Parquet, joblib to
  pickle), you update one file, not six.
- Responsibility (separation of concerns): utils.py owns ONLY file system interactions.
  It does not transform data, train models, or make decisions about the pipeline logic.
- Pipeline contract (inputs and outputs):
    Inputs : file paths (pathlib.Path objects) and data/model objects
    Outputs: DataFrames loaded from disk, or None (side-effect: files written to disk)

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import joblib
import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: a pathlib.Path pointing to the CSV file to read
    Outputs:
    - pd.DataFrame containing the contents of the CSV file
    Why this contract matters for reliable ML delivery:
    - A single load function means all CSV reads go through the same code path.
      You can add encoding handling, dtype enforcement, or audit logging here once
      and every caller benefits automatically.
    """
    print(f"[utils] Loading CSV from: {filepath}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust read parameters to match your file.
    # Why: Different datasets may need encoding, separators, or dtypes specified.
    # Examples:
    # 1. pd.read_csv(filepath, encoding="latin-1")  — for non-UTF-8 files
    # 2. pd.read_csv(filepath, sep=";")             — for semicolon-delimited files
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    df = pd.read_csv(filepath)
    print(f"[utils]   Loaded {len(df)} rows x {len(df.columns)} columns.")  # TODO: replace with logging later
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df      : the DataFrame to write to disk
    - filepath: a pathlib.Path for the destination CSV file
    Outputs:
    - None (side-effect: CSV file written to disk)
    Why this contract matters for reliable ML delivery:
    - Centralising CSV writes means every artifact is saved consistently
      (no index column, predictable encoding). It also auto-creates the parent
      directory, preventing FileNotFoundError in fresh environments or CI.
    """
    print(f"[utils] Saving CSV to: {filepath}")  # TODO: replace with logging later

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust save parameters if needed.
    # Why: Some workflows need the index preserved, or a different separator.
    # Examples:
    # 1. df.to_csv(filepath, index=True)   — keep row index in the file
    # 2. df.to_csv(filepath, sep=";")      — semicolon-delimited output
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    df.to_csv(filepath, index=False)
    print(f"[utils]   Saved {len(df)} rows to {filepath}")  # TODO: replace with logging later


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model   : a fitted scikit-learn Pipeline (or any joblib-serialisable object)
    - filepath: a pathlib.Path for the destination .joblib file
    Outputs:
    - None (side-effect: model serialised to disk)
    Why this contract matters for reliable ML delivery:
    - Saving the full Pipeline (preprocessor + estimator together) guarantees that
      inference always applies the exact same transformations the model was trained
      with. Saving just the estimator is a common mistake that causes silent bugs.
    """
    print(f"[utils] Saving model to: {filepath}")  # TODO: replace with logging later

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust the joblib.dump parameters if needed.
    # Why: compress=3 is a good default for smaller file sizes on disk.
    # Examples:
    # 1. joblib.dump(model, filepath, compress=3)  — compressed save
    # 2. joblib.dump(model, filepath, compress=0)  — no compression, faster write
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    joblib.dump(model, filepath)
    print(f"[utils]   Model saved to {filepath}")  # TODO: replace with logging later


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: a pathlib.Path pointing to a .joblib model file
    Outputs:
    - The deserialised model object (typically a fitted sklearn Pipeline)
    Why this contract matters for reliable ML delivery:
    - A dedicated load function is the correct place to add version checks or
      compatibility warnings in the future (e.g. warn if model was trained on a
      different scikit-learn version). Calling joblib.load directly everywhere
      makes those checks impossible to add consistently.
    """
    print(f"[utils] Loading model from: {filepath}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add a file existence check if needed for your deployment.
    # Why: In production, a missing model file should produce a clear error message,
    #      not a cryptic FileNotFoundError deep in joblib.
    # Examples:
    # 1. if not filepath.exists(): raise FileNotFoundError(f"Model not found: {filepath}")
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    model = joblib.load(filepath)
    print(f"[utils]   Model loaded from {filepath}")  # TODO: replace with logging later
    return model
