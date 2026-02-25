"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python -m src.main
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: main.py is the top-level orchestrator.
  It wires together every stage of the pipeline in one readable, linear script so
  any team member can understand the full flow at a glance.
- Responsibility (separation of concerns): main.py owns ONLY the order-of-operations
  and configuration. It delegates all real work to the src sub-modules. Think of it
  as a table of contents for your pipeline.
- Pipeline contract (inputs and outputs):
    Inputs : raw CSV path defined in SETTINGS
    Outputs: data/processed/clean.csv, models/model.joblib, reports/predictions.csv

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

# ===========================================================================
# 1. IMPORTS
# ===========================================================================
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
from src.utils import save_csv, save_model

if __name__ == "__main__":
    print("Pipeline not implemented yet. HAhaahh")