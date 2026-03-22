"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow
        (Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python -m src.main

TODO: Any temporary or hardcoded variable or parameter will be imported from
    config.yml in a later session
"""

# ===========================================================================
# 1. IMPORTS
# ===========================================================================
# Standard Library Imports
from pathlib import Path
import logging

# Third-party Imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Local Module Imports
from logger import configure_logging
from load_data import load_raw_data
from clean_data import clean_dataframe
from validate import validate_dataframe
from features import get_feature_preprocessor
from utils import save_csv, save_model

# Get kmeans functions
from kmeans.kmeans_evaluate import evaluate_kmeans_model
from kmeans.kmeans_infer import run_kmeans_inference
from kmeans.kmeans_train import train_kmeans_model

# Get logitic regression functions
from logit_regression.logit_train import train_logit_model
from logit_regression.logit_evaluate import evaluate_logit_model
from logit_regression.logit_infer import run_logit_inference

# Get decision tree functions
from dtrees.dtrees_train import train_dtrees_model
from dtrees.dtrees_eval import evaluate_dtrees_model
from dtrees.dtrees_infer import run_dtrees_inference

logger = logging.getLogger(__name__)

# ===========================================================================
# 2. CONFIGURATION  —  SETTINGS BRIDGE
# ===========================================================================
SETTINGS = {
    "raw_data_path":       Path("data/raw/train.csv"),
    "processed_data_path": Path("data/processed/clean.csv"),
    "kmeans_model_path":   Path("models/kmeans_model.joblib"),
    "logit_model_path":    Path("models/logit_model.joblib"),
    "dtrees_model_path":    Path("models/dtrees_model.joblib"),
    "kmeans_predictions_path":    Path("reports/kmeans_predictions.csv"),
    "logit_predictions_path":    Path("reports/logit_predictions.csv"),
    "dtrees_predictions_path":    Path("reports/dtrees_predictions.csv"),
    "target_column": "Heart Disease",
    "problem_type": "classification",
    "test_size":    0.2,
    "random_state": 42,
    "k_means_bins": 3,
    "max_iterations": 2000,
    "features": {
        "quantile_bin":        [],
        "categorical_onehot":  [],
        "numeric_passthrough": ["id", "Sex", "FBS over 120",
                                "EKG results", "Exercise angina",
                                "ST depression"],
        "ordinal_encode": ["Chest pain type", "Number of vessels fluro",
                           "Thallium", "Slope of ST"],
        "min_max_scaler": ["Age", "BP", "Cholesterol", "Max HR"],
        "n_bins": 3
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
    """
    # -------------------------------------------------------------------
    # STEP 0: Configure logging and ensure output directories exist
    # -------------------------------------------------------------------
    configure_logging(log_level="INFO", log_file=Path("logs/pipeline.log"))
    logger.info("Creating output directories if they do not exist...")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # STEP 1: Load raw data
    # -------------------------------------------------------------------
    logger.info("Step 1 — Loading raw data...")
    df_raw = load_raw_data(SETTINGS["raw_data_path"])

    # -------------------------------------------------------------------
    # STEP 2: Clean data
    # -------------------------------------------------------------------
    logger.info("Step 2 — Cleaning data...")
    df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])

    # -------------------------------------------------------------------
    # STEP 3: Save processed CSV
    # -------------------------------------------------------------------
    logger.info("Step 3 — Saving processed CSV...")
    save_csv(df_clean, SETTINGS["processed_data_path"])

    # -------------------------------------------------------------------
    # STEP 4: Validate cleaned data
    # -------------------------------------------------------------------
    logger.info("Step 4 — Validating cleaned data...")
    required_cols = (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
        + SETTINGS["features"]["ordinal_encode"]
        + SETTINGS["features"]["min_max_scaler"]
        + [SETTINGS["target_column"]]
    )
    validate_dataframe(df_clean, required_columns=required_cols)

    # -------------------------------------------------------------------
    # STEP 5: Train / test split  (MUST happen BEFORE feature fitting)
    # -------------------------------------------------------------------
    logger.info("Step 5 — Splitting into train and test sets...")
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
            logger.info(
                "Stratified split failed (%s). "
                "Falling back to random split.", e
            )
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

    logger.info("Train size: %d rows | Test size: %d rows",
                len(X_train), len(X_test))

    # -------------------------------------------------------------------
    # STEP 6: Fail-fast feature checks
    # -------------------------------------------------------------------
    logger.info("Step 6 — Verifying feature columns exist in training data")
    all_feature_cols = (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
        + SETTINGS["features"]["ordinal_encode"]
        + SETTINGS["features"]["min_max_scaler"]
    )
    missing_cols = [c for c in all_feature_cols if c not in X_train.columns]
    if missing_cols:
        raise ValueError(
                    f"[main] The following configured feature columns are "
                    f"missing from the data: {missing_cols}\n"
                    "Update the 'features' section in SETTINGS to match "
                    "your actual column names."
                )

    # Check that quantile_bin columns are numeric
    for col in SETTINGS["features"]["quantile_bin"]:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            raise TypeError(
                    f"[main] Column '{col}' is listed under "
                    "'quantile_bin' but is not numeric. "
                    "Only numeric columns can be binned."
                )

    # -------------------------------------------------------------------
    # STEP 7: Build feature preprocessor (ColumnTransformer recipe only —
    #          no fitting here, fitting happens inside train_model)
    # -------------------------------------------------------------------
    logger.info("Step 7 — Building feature preprocessor recipe...")
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        min_max_cols=SETTINGS["features"]["min_max_scaler"],
        ordinal_encode_cols=SETTINGS["features"]["ordinal_encode"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],

        n_bins=SETTINGS["features"]["n_bins"],
    )

    # -------------------------------------------------------------------
    # STEP 8: Train model (Pipeline: preprocess + estimator, fit on train)
    # -------------------------------------------------------------------
    logger.info("Step 8 — Training model...")
    kmeans_model = train_kmeans_model(X_train=X_train,
                                      preprocessor=preprocessor,
                                      n_clusters=SETTINGS["k_means_bins"])

    # dtrees train model
    dtrees_model = train_dtrees_model(X_train=X_train, y_train=y_train,
                                      preprocessor=preprocessor,
                                      problem_type=SETTINGS["problem_type"],)

    # logit train model
    logit_model = train_logit_model(X_train, y_train, preprocessor,
                                    SETTINGS["problem_type"],
                                    SETTINGS["random_state"],
                                    SETTINGS["max_iterations"])

    # -------------------------------------------------------------------
    # STEP 9: Save trained model artifact
    # -------------------------------------------------------------------
    logger.info("Step 9 — Saving model artifact...")
    save_model(kmeans_model, SETTINGS["kmeans_model_path"])
    save_model(dtrees_model, SETTINGS["dtrees_model_path"])
    save_model(logit_model, SETTINGS["logit_model_path"])

    # -------------------------------------------------------------------
    # STEP 10: Evaluate on held-out test set
    # -------------------------------------------------------------------
    logger.info("Step 10 — Evaluating model on test set...")

    # Evaluate kmeans model
    kmeans_metric_value = evaluate_kmeans_model(model=kmeans_model,
                                                X_test=X_test)
    metric_label = ("RMSE" if SETTINGS["problem_type"] == "regression"
                    else "F1 (weighted)")
    logger.info("Kmeans Metrics %s: %.4f", metric_label, kmeans_metric_value)

    # Evaluate dtrees model
    dtrees_metrics = evaluate_dtrees_model(model=dtrees_model, X_test=X_test,
                                           y_test=y_test,
                                           prob_type=SETTINGS['problem_type'])
    logger.info("Decision Tree Metrics %s: %s", metric_label, dtrees_metrics)

    # Evaluate logit model
    logit_metrics = evaluate_logit_model(model=logit_model, X_test=X_test,
                                         y_test=y_test,
                                         prob_type=SETTINGS['problem_type'])
    logger.info("Logistic Regression Metrics %s: %s",
                metric_label, logit_metrics)

    # -------------------------------------------------------------------
    # STEP 11: Run inference on example data (first 5 rows of test set)
    # -------------------------------------------------------------------
    logger.info("Step 11 — Running inference on example rows...")
    # Kmeans inference example
    X_example = X_test.head(5)
    kmeans_predictions_df = run_kmeans_inference(model=kmeans_model,
                                                 X_infer=X_example)
    # Dtrees inference example
    dtrees_predictions_df = run_dtrees_inference(model=dtrees_model,
                                                 X_infer=X_example)
    # Logit inference example
    logit_predictions_df = run_logit_inference(model=logit_model,
                                               X_infer=X_example,
                                               optimal_threshold=logit_metrics["optimal_threshold"])

    # -------------------------------------------------------------------
    # STEP 12: Save predictions artifact
    # -------------------------------------------------------------------
    logger.info("Step 12 — Saving predictions CSV...")
    save_csv(kmeans_predictions_df, SETTINGS["kmeans_predictions_path"])
    save_csv(dtrees_predictions_df, SETTINGS["dtrees_predictions_path"])
    save_csv(logit_predictions_df, SETTINGS["logit_predictions_path"])

    logger.info("Pipeline complete!")
    logger.info("  Processed data : %s", SETTINGS['processed_data_path'])
    logger.info("  Model artifact : %s", SETTINGS['logit_model_path'])
    logger.info("  Predictions    : %s", SETTINGS['kmeans_predictions_path'])


if __name__ == "__main__":
    main()
