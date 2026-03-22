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

# Utils imports
from utils import save_csv, save_model
from utils import load_config, require_section, require_str 
from utils import require_float, require_int, require_list
from utils import resolve_repo_path, normalize_problem_type

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
# 2. MAIN PIPELINE
# ===========================================================================
def main():
    """
    Inputs:
    -  Config.yaml file for configuration of model settings
    Outputs:
    - data/processed/clean.csv   — cleaned dataset written to disk
    - models/model.joblib        — trained sklearn Pipeline written to disk
    - reports/predictions.csv   — inference output written to disk
    """
    project_root = Path(__file__).resolve().parents[1]

    # -----------------------------
    # Load and validate config.yaml
    # -----------------------------
    cfg = load_config(project_root / "config.yaml")

    # Load required sections from config file
    paths_cfg = require_section(cfg, "paths")
    problem_cfg = require_section(cfg, "problem")
    split_cfg = require_section(cfg, "split")
    features_cfg = require_section(cfg, "features")
    training_cfg = require_section(cfg, "training")
    logging_cfg = require_section(cfg, "logging")

    # Problem config
    PROBLEM_TYPE = normalize_problem_type(
            require_str(problem_cfg, "problem_type"))
    TARGET_COLUMN = require_str(problem_cfg, "target_column")
    if PROBLEM_TYPE not in {"classification", "regression"}:
        raise ValueError(
            "config.yaml: problem.problem_type must be 'classification' or 'regression'"
        )

    # Config data and log paths as well as log level
    LOG_FILE_PATH = resolve_repo_path(
        project_root, require_str(paths_cfg, "log_file"), ensure_parent=True)
    LOG_LEVEL = require_str(logging_cfg, "level")
    RAW_DATA_PATH = resolve_repo_path(
        project_root, require_str(paths_cfg, "raw_data"))
    PROCESS_DATA_PATH = resolve_repo_path(
        project_root, require_str(paths_cfg, "processed_data"), ensure_parent=True)
    
    # train and split values
    MAX_ITERATIONS = require_int(training_cfg, "max_iterations")
    K_MEANS_BINS = require_int(training_cfg, "k_means_bins")
    RANDOM_STATE = require_int(split_cfg, "random_state")
    TEST_SIZE = require_float(split_cfg, "test_size")
    
    # Model paths
    KMEANS_MODEL_PATH = resolve_repo_path(
        project_root, require_str(paths_cfg, "kmeans_model"), ensure_parent=True)
    DTREES_MODEL_PATH = resolve_repo_path(
        project_root, require_str(paths_cfg, "dtrees_model"))
    LOGIT_MODEL_PATH = resolve_repo_path(
        project_root, require_str(paths_cfg, "logit_model"))
    
    # model predictions
    KMEANS_PRED_PATH = resolve_repo_path(
        project_root, require_str(paths_cfg, "kmeans_predictions"), ensure_parent=True)
    DTREES_PRED_PATH = resolve_repo_path(
        project_root, require_str(paths_cfg, "dtrees_predictions"))
    LOGIT_PRED_PATH = resolve_repo_path(
        project_root, require_str(paths_cfg, "logit_predictions"))
    
    # Feature columns
    QUANTILE_BINS_COLS = require_list(features_cfg, "quantile_bin")
    CATEGORICAL_ONEHOT_COLS = require_list(
        features_cfg, "categorical_onehot")
    NUMERIC_PASSTHROUGH_COLS = require_list(
        features_cfg, "numeric_passthrough")
    ORDINAL_ENCODE_COLS = require_list(features_cfg, "ordinal_encode")
    MIN_MAX_SCALAR_COLS = require_list(features_cfg, "min_max_scaler")
    N_BINS = require_int(features_cfg, "n_bins")

    # -------------------------------------------------------------------
    # STEP 0: Configure logging and ensure output directories exist
    # -------------------------------------------------------------------
    configure_logging(log_level=LOG_LEVEL, log_file=LOG_FILE_PATH)
    logger.info("Creating output directories if they do not exist...")

    # -------------------------------------------------------------------
    # STEP 1: Load raw data
    # -------------------------------------------------------------------
    logger.info("Step 1 — Loading raw data...")

    df_raw = load_raw_data(RAW_DATA_PATH)

    # -------------------------------------------------------------------
    # STEP 2: Clean data
    # -------------------------------------------------------------------
    logger.info("Step 2 — Cleaning data...")
    df_clean = clean_dataframe(df_raw, target_column=TARGET_COLUMN)

    # -------------------------------------------------------------------
    # STEP 3: Save processed CSV
    # -------------------------------------------------------------------
    logger.info("Step 3 — Saving processed CSV...")
    save_csv(df_clean, PROCESS_DATA_PATH)

    # -------------------------------------------------------------------
    # STEP 4: Validate cleaned data
    # -------------------------------------------------------------------
    logger.info("Step 4 — Validating cleaned data...")
    required_cols = (
        QUANTILE_BINS_COLS
        + CATEGORICAL_ONEHOT_COLS
        + NUMERIC_PASSTHROUGH_COLS
        + ORDINAL_ENCODE_COLS
        + MIN_MAX_SCALAR_COLS
        + [TARGET_COLUMN]
    )
    validate_dataframe(df_clean, required_columns=required_cols)

    # -------------------------------------------------------------------
    # STEP 5: Train / test split  (MUST happen BEFORE feature fitting)
    # -------------------------------------------------------------------
    logger.info("Step 5 — Splitting into train and test sets...")
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]

    stratify_arg = None
    if PROBLEM_TYPE == "classification":
        try:
            stratify_arg = y
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=stratify_arg,
            )
        except ValueError as e:
            logger.info(
                "Stratified split failed (%s). "
                "Falling back to random split.", e
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    logger.info("Train size: %d rows | Test size: %d rows",
                len(X_train), len(X_test))

    # -------------------------------------------------------------------
    # STEP 6: Fail-fast feature checks
    # -------------------------------------------------------------------
    logger.info("Step 6 — Verifying feature columns exist in training data")
    all_feature_cols = (
        QUANTILE_BINS_COLS
        + CATEGORICAL_ONEHOT_COLS
        + NUMERIC_PASSTHROUGH_COLS
        + ORDINAL_ENCODE_COLS
        + MIN_MAX_SCALAR_COLS
    )
    missing_cols = [c for c in all_feature_cols if c not in X_train.columns]
    if missing_cols:
        raise ValueError(
                    f"[main] The following configured feature columns are "
                    f"missing from the data: {missing_cols}\n"
                    "Update the 'features' section in config to match "
                    "your actual column names."
                )

    # Check that quantile_bin columns are numeric
    for col in QUANTILE_BINS_COLS:
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
        quantile_bin_cols=QUANTILE_BINS_COLS,
        categorical_onehot_cols=CATEGORICAL_ONEHOT_COLS,
        min_max_cols=MIN_MAX_SCALAR_COLS,
        ordinal_encode_cols=ORDINAL_ENCODE_COLS,
        numeric_passthrough_cols=NUMERIC_PASSTHROUGH_COLS,

        n_bins=N_BINS,
    )

    # -------------------------------------------------------------------
    # STEP 8: Train model (Pipeline: preprocess + estimator, fit on train)
    # -------------------------------------------------------------------
    logger.info("Step 8 — Training model...")
    kmeans_model = train_kmeans_model(X_train=X_train,
                                      preprocessor=preprocessor,
                                      n_clusters=K_MEANS_BINS)

    # dtrees train model
    dtrees_model = train_dtrees_model(X_train=X_train, y_train=y_train,
                                      preprocessor=preprocessor,
                                      problem_type=PROBLEM_TYPE)

    # logit train model
    logit_model = train_logit_model(X_train, y_train, preprocessor,
                                    PROBLEM_TYPE,
                                    RANDOM_STATE,
                                    MAX_ITERATIONS)

    # -------------------------------------------------------------------
    # STEP 9: Save trained model artifact
    # -------------------------------------------------------------------
    logger.info("Step 9 — Saving model artifact...")
    save_model(kmeans_model, KMEANS_MODEL_PATH)
    save_model(dtrees_model, DTREES_MODEL_PATH)
    save_model(logit_model, LOGIT_MODEL_PATH)

    # -------------------------------------------------------------------
    # STEP 10: Evaluate on held-out test set
    # -------------------------------------------------------------------
    logger.info("Step 10 — Evaluating model on test set...")

    # Evaluate kmeans model
    kmeans_metric_value = evaluate_kmeans_model(model=kmeans_model,
                                                X_test=X_test)
    metric_label = ("RMSE" if PROBLEM_TYPE == "regression"
                    else "F1 (weighted)")
    logger.info("Kmeans Metrics %s: %.4f", metric_label, kmeans_metric_value)

    # Evaluate dtrees model
    dtrees_metrics = evaluate_dtrees_model(model=dtrees_model, X_test=X_test,
                                           y_test=y_test,
                                           prob_type=PROBLEM_TYPE)
    logger.info("Decision Tree Metrics %s: %s", metric_label, dtrees_metrics)

    # Evaluate logit model
    logit_metrics = evaluate_logit_model(model=logit_model, X_test=X_test,
                                         y_test=y_test,
                                         prob_type=PROBLEM_TYPE)
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
    save_csv(kmeans_predictions_df, KMEANS_PRED_PATH)
    save_csv(dtrees_predictions_df, DTREES_PRED_PATH)
    save_csv(logit_predictions_df, LOGIT_PRED_PATH)

    logger.info("Pipeline complete!")
    logger.info("  Processed data : %s", PROCESS_DATA_PATH)
    logger.info("  Model artifact : %s", LOGIT_MODEL_PATH)
    logger.info("  Predictions    : %s", KMEANS_PRED_PATH)


if __name__ == "__main__":
    main()
