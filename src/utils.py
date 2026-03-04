from pathlib import Path

import joblib
import pandas as pd
import csv


def _check_path_exists(filepath: Path):
    """Check if the given file path exists on disk.
    If not, raise a FileNotFoundError with a clear message.\n
    Inputs:
    - filepath: a pathlib.Path pointing to the file to check\n
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")


def _get_csv_delimiter(filepath: Path) -> str:
    """Get the delimiter used in the CSV file by reading a sample and
    using csv.Sniffer.\n
    Inputs:
    - filepath: a pathlib.Path pointing to the CSV file to check\n
    Outputs:
    - str representing the delimiter used in the CSV file
    """
    with open(filepath, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        return dialect.delimiter


def _make_parent_dir(filepath: Path) -> None:
    """Create the parent directory for the given file path if it
    doesn't already exist.\n
    Inputs:
    - filepath: a pathlib.Path pointing to the file whose parent
      directory should be created\n
    Outputs:
    - None (side-effect: parent directory created on disk if it
      didn't exist)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load the csv file from the given filepath and return it as a
    pandas DataFrame.\n
    Inputs:
    - filepath: a pathlib.Path pointing to the CSV file to read\n
    Outputs:
    - pd.DataFrame containing the contents of the CSV file
    """

    # Make sure file exists
    _check_path_exists(filepath)
    msg = f"[utils] Loading CSV from: {filepath}"
    print(msg)  # TODO: replace with logging later

    df = pd.read_csv(filepath, sep=_get_csv_delimiter(filepath))
    msg = (f"[utils]   Loaded {len(df)} rows x "
           f"{len(df.columns)} columns.")
    print(msg)  # TODO: replace with logging later
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Save the given DataFrame to a CSV file at the specified filepath.
    If the parent directory doesn't exist, it will be created.\n
    Inputs:
    - df      : the DataFrame to write to disk
    - filepath: a pathlib.Path for the destination CSV file\n
    Outputs:
    - None (side-effect: CSV file written to disk)
    """
    _make_parent_dir(filepath)
    msg = f"[utils] Saving CSV to: {filepath}"
    print(msg)  # TODO: replace with logging later

    df.to_csv(filepath, index=False, sep=",")
    msg = f"[utils]   Saved {len(df)} rows to {filepath}"
    print(msg)  # TODO: replace with logging later


def save_model(model, filepath: Path) -> None:
    """Save the given model object to a .joblib file at the specified
    filepath. If the parent directory doesn't exist, it will be created.\n
    Inputs:
    - model   : a fitted scikit-learn Pipeline (or any joblib-serialisable
    object)
    - filepath: a pathlib.Path for the destination .joblib file\n
    Outputs:
    - None (side-effect: model serialised to disk)
    """
    _make_parent_dir(filepath)
    msg = f"[utils] Saving model to: {filepath}"
    print(msg)  # TODO: replace with logging later

    joblib.dump(model, filepath, compress=3)
    msg = f"[utils]   Model saved to {filepath}"
    print(msg)  # TODO: replace with logging later


def load_model(filepath: Path):
    """Load and return the model object from the specified .joblib file.\n
    Inputs:
    - filepath: a pathlib.Path pointing to a .joblib model file\n
    Outputs:
    - The deserialised model object (typically a fitted sklearn Pipeline)
    """
    _check_path_exists(filepath)
    msg = f"[utils] Loading model from: {filepath}"
    print(msg)  # TODO: replace with logging later

    model = joblib.load(filepath)
    msg = f"[utils]   Model loaded from {filepath}"
    print(msg)  # TODO: replace with logging later
    return model
