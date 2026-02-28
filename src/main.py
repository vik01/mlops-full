"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python -m src.main

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
# from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor

# Get kmeans functions
from kmeans.evaluate import evaluate_kmeans_model
from kmeans.infer import run_kmeans_inference
from kmeans.train import train_kmeans_model

from src.utils import save_csv, save_model

# ===========================================================================
# 2. CONFIGURATION  —  SETTINGS BRIDGE
# ===========================================================================
SETTINGS = {
    "is_example_config": False,
    "raw_data_path":       Path("data/raw/train.csv"),
    "processed_data_path": Path("data/processed/clean.csv"),
    "model_path":          Path("models/model.joblib"),
    "predictions_path":    Path("reports/predictions.csv"),
    "target_column": "Heart Disease",
    "problem_type": "classification",
    "test_size":    0.2,
    "random_state": 42,
    "features": {
        "quantile_bin":        ["ST depression"],
        "categorical_onehot":  [],
        "numeric_passthrough": ["id", "Sex", "FBS over 120", 
                                "EKG results", "Exercise angina"],
        "ordinal_encode": ["Chest pain type", "Number of vessels fluro", 
                           "Thallium", "Slope of ST"],
        "min_max_scaler": ["Age", "BP", "Cholesterol", "Max HR"],
        "n_bins": 5
    }
}

# ===========================================================================
# 3. MAIN PIPELINE
# ===========================================================================
def main():
  """
  Inputs:
  - SETTINGS dictionary defined at module level (acts as config bridge)
  Outputs:
  - data/processed/clean.csv   — cleaned dataset written to disk
  - models/model.joblib        — trained sklearn Pipeline written to disk
  - reports/predictions.csv   — inference output written to disk
  Why this contract matters for reliable ML delivery:
  - A single entry point means any engineer can reproduce the full pipeline
    with one command, reducing 'works on my machine' failures and enabling
    CI/CD automation.
  """
    # -------------------------------------------------------------------
  # STEP 0: Ensure output directories exist
  # -------------------------------------------------------------------
  print("\n[main] Creating output directories if they do not exist...")  # TODO: replace with logging later
  Path("data/processed").mkdir(parents=True, exist_ok=True)
  Path("models").mkdir(parents=True, exist_ok=True)
  Path("reports").mkdir(parents=True, exist_ok=True)

  if SETTINGS["is_example_config"]:
    print(  # TODO: replace with logging later
            "\n" + "=" * 70 + "\n"
            "WARNING: is_example_config is True.\n"
            "You are running with DUMMY settings and a generated sample CSV.\n"
            "Update SETTINGS in src/main.py before using real data!\n"
            + "=" * 70 + "\n"
    )
  
  # -------------------------------------------------------------------
  # STEP 1: Load raw data
  # -------------------------------------------------------------------
  print("[main] Step 1 — Loading raw data...")  # TODO: replace with logging later
  # TERESA: You for your load_raw_data() function. You need to take in a Path Object.
  df_raw = load_raw_data(SETTINGS["raw_data_path"])


  # -------------------------------------------------------------------
  # STEP 2: Clean data
  # -------------------------------------------------------------------
  print("[main] Step 2 — Cleaning data...")  # TODO: replace with logging later
  # ALESSANDRO: You for your clean_dataframe() function. You need to take in the raw dataframe and target column name (string).
  # df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])
  df_clean = pd.DataFrame()

  # -------------------------------------------------------------------
  # STEP 3: Save processed CSV
  # -------------------------------------------------------------------
  print("[main] Step 3 — Saving processed CSV...")  # TODO: replace with logging later
  save_csv(df_clean, SETTINGS["processed_data_path"])

  # -------------------------------------------------------------------
  # STEP 4: Validate cleaned data
  # -------------------------------------------------------------------
  print("[main] Step 4 — Validating cleaned data...")  # TODO: replace with logging later
  required_cols = (
      SETTINGS["features"]["quantile_bin"]
      + SETTINGS["features"]["categorical_onehot"]
      + SETTINGS["features"]["numeric_passthrough"]
      + [SETTINGS["target_column"]]
  )
  validate_dataframe(df_clean, required_columns=required_cols)

  # -------------------------------------------------------------------
  # STEP 5: Train / test split  (MUST happen BEFORE feature fitting)
  # -------------------------------------------------------------------
  print("[main] Step 5 — Splitting into train and test sets...")  # TODO: replace with logging later
  X = df_clean.drop(columns=[SETTINGS["target_column"]])
  y = df_clean[SETTINGS["target_column"]]

  stratify_arg = None
  if SETTINGS["problem_type"] == "classification":
      try:
          stratify_arg = y
          X_train, X_test, y_train, y_test = train_test_split(
              X, y,
              test_size=SETTINGS["test_size"],
              random_state=SETTINGS["random_state"],
              stratify=stratify_arg,
          )
      except ValueError as e:
          print(f"[main] Stratified split failed ({e}). Falling back to random split.")  # TODO: replace with logging later
          X_train, X_test, y_train, y_test = train_test_split(
              X, y,
              test_size=SETTINGS["test_size"],
              random_state=SETTINGS["random_state"],
          )
  else:
      X_train, X_test, y_train, y_test = train_test_split(
          X, y,
          test_size=SETTINGS["test_size"],
          random_state=SETTINGS["random_state"],
      )

  print(f"[main]   Train size: {len(X_train)} rows | Test size: {len(X_test)} rows")  # TODO: replace with logging later

  # -------------------------------------------------------------------
  # STEP 6: Fail-fast feature checks
  # -------------------------------------------------------------------
  print("[main] Step 6 — Verifying feature columns exist in training data...")  # TODO: replace with logging later
  all_feature_cols = (
      SETTINGS["features"]["quantile_bin"]
      + SETTINGS["features"]["categorical_onehot"]
      + SETTINGS["features"]["numeric_passthrough"]
  )
  missing_cols = [c for c in all_feature_cols if c not in X_train.columns]
  if missing_cols:
      raise ValueError(
          f"[main] The following configured feature columns are missing from the data: {missing_cols}\n"
          "Update the 'features' section in SETTINGS to match your actual column names."
      )

  # Check that quantile_bin columns are numeric
  for col in SETTINGS["features"]["quantile_bin"]:
      if not pd.api.types.is_numeric_dtype(X_train[col]):
          raise TypeError(
              f"[main] Column '{col}' is listed under 'quantile_bin' but is not numeric. "
              "Only numeric columns can be binned."
          )
  
  # -------------------------------------------------------------------
  # STEP 7: Build feature preprocessor (ColumnTransformer recipe only —
  #          no fitting here, fitting happens inside train_model)
  # -------------------------------------------------------------------
  print("[main] Step 7 — Building feature preprocessor recipe...")  # TODO: replace with logging later
  preprocessor = get_feature_preprocessor(
      quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
      categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
      numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
      n_bins=SETTINGS["features"]["n_bins"],
  )

  # --------------------------------------------------------
  # START STUDENT CODE
  # --------------------------------------------------------
  # TODO_STUDENT: Customise the SETTINGS dictionary above and the
  #               feature lists to match your real dataset columns.
  # Why: Every dataset has different column names, dtypes, and split
  #      strategies. This is where your domain knowledge drives the config.
  # Examples:
  # 1. Change "target_column" to the name of your label column.
  # 2. Move columns between quantile_bin, categorical_onehot, and
  #    numeric_passthrough depending on their type and cardinality.
  #
  # Optional forcing function (leave commented)
  # raise NotImplementedError("Student: You must implement this logic to proceed!")
  #
  # Placeholder (Remove this after implementing your code):
  print("Warning: Student has not implemented this section yet")
  # --------------------------------------------------------
  # END STUDENT CODE
  # --------------------------------------------------------

  # -------------------------------------------------------------------
  # STEP 8: Train model (Pipeline: preprocess + estimator, fit on train)
  # -------------------------------------------------------------------
  print("[main] Step 8 — Training model...")  # TODO: replace with logging later
#   model = train_kmeans_model(
#       X_train=X_train,
#       y_train=y_train,
#       preprocessor=preprocessor,
#       problem_type=SETTINGS["problem_type"],
#   )
  model = train_kmeans_model(X_train=X_train, preprocessor=preprocessor, n_clusters=SETTINGS["n_bins"],)

  # -------------------------------------------------------------------
  # STEP 9: Save trained model artifact
  # -------------------------------------------------------------------
  print("[main] Step 9 — Saving model artifact...")  # TODO: replace with logging later
  save_model(model, SETTINGS["model_path"])

  # -------------------------------------------------------------------
  # STEP 10: Evaluate on held-out test set
  # -------------------------------------------------------------------
  print("[main] Step 10 — Evaluating model on test set...")  # TODO: replace with logging later
#   metric_value = evaluate_model(
#       model=model,
#       X_test=X_test,
#       y_test=y_test,
#       problem_type=SETTINGS["problem_type"],
#   )
  metric_value = evaluate_kmeans_model(model=model,X_test=X_test)
  metric_label = "RMSE" if SETTINGS["problem_type"] == "regression" else "F1 (weighted)"
  print(f"[main]   {metric_label}: {metric_value:.4f}")  # TODO: replace with logging later

  # -------------------------------------------------------------------
  # STEP 11: Run inference on example data (first 5 rows of test set)
  # -------------------------------------------------------------------
  print("[main] Step 11 — Running inference on example rows...")  # TODO: replace with logging later
  X_example = X_test.head(5)
  predictions_df = run_kmeans_inference(model=model, X_infer=X_example)

  # -------------------------------------------------------------------
  # STEP 12: Save predictions artifact
  # -------------------------------------------------------------------
  print("[main] Step 12 — Saving predictions CSV...")  # TODO: replace with logging later
  save_csv(predictions_df, SETTINGS["predictions_path"])

  print("\n[main] Pipeline complete!")  # TODO: replace with logging later
  print(f"  Processed data : {SETTINGS['processed_data_path']}")
  print(f"  Model artifact : {SETTINGS['model_path']}")
  print(f"  Predictions    : {SETTINGS['predictions_path']}")


    

if __name__ == "__main__":
    print("Pipeline not implemented yet. HAhaahh")
    # main()