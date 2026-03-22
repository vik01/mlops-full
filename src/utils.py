# Standard Library Imports
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Third-party Imports
import joblib
import yaml
import pandas as pd

logger = logging.getLogger(__name__)


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
    logger.info("Loading CSV from: %s", filepath)

    df = pd.read_csv(filepath, sep=_get_csv_delimiter(filepath))
    logger.info("Loaded %d rows x %d columns.", len(df), len(df.columns))
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
    logger.info("Saving CSV to: %s", filepath)

    df.to_csv(filepath, index=False, sep=",")
    logger.info("Saved %d rows to %s", len(df), filepath)


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
    logger.info("Saving model to: %s", filepath)

    joblib.dump(model, filepath, compress=3)
    logger.info("Model saved to %s", filepath)


def load_model(filepath: Path):
    """Load and return the model object from the specified .joblib file.\n
    Inputs:
    - filepath: a pathlib.Path pointing to a .joblib model file\n
    Outputs:
    - The deserialised model object (typically a fitted sklearn Pipeline)
    """
    _check_path_exists(filepath)
    logger.info("Loading model from: %s", filepath)

    model = joblib.load(filepath)
    logger.info("Model loaded from %s", filepath)
    return model


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration from disk

    Why this exists
    - Centralizing config loading prevents "config drift" where different modules parse YAML differently
    - Fail fast with clear messages when config.yaml is missing or malformed
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            "config.yaml must parse into a dictionary at the top level")

    return cfg


def require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Enforce a required top-level config section

    Why this exists
    - This produces an actionable error tied to config.yaml structure
    """
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ValueError(
            f"config.yaml must contain a top-level '{section}' mapping")
    return value


def require_str(section: Dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.yaml: '{key}' must be a non-empty string")
    return value.strip()


def require_float(section: Dict[str, Any], key: str) -> float:
    value = section.get(key)
    try:
        return float(value) # type: ignore
    except Exception as e:
        raise ValueError(
            f"config.yaml: '{key}' must be a number. Got '{value}'") from e
    

def require_int(section: Dict[str, Any], key: str) -> int:
    value = section.get(key)
    try:
        return int(value) # type: ignore
    except Exception as e:
        raise ValueError(
            f"config.yaml: '{key}' must be an integer. Got '{value}'") from e


def require_list(section: Dict[str, Any], key: str) -> List[str]:
    value = section.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            f"config.yaml: '{key}' must be a list. Got type={type(value)}")
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def normalize_problem_type(problem_type: Optional[str]) -> str:
    return (problem_type or "").strip().lower()


def resolve_repo_path(project_root: Path, relative_path: str, 
                      ensure_parent: bool = False) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("config.yaml: path values must be non-empty strings")
    resolved = project_root / relative_path.strip()
    if ensure_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
