"""
Module: Decision Tree Inference
---------------------------------
Role: Apply the trained Decision Tree to new, unlabelled data.
Input: Fitted Pipeline + X_infer (DataFrame of features, no target column).
Output: DataFrame with a single column "prediction", preserving input index.

Educational Goal:
- Why this module exists in an MLOps system:
    Inference is a separate concern from training and evaluation. In production
    this function is what gets called by an API endpoint for real-time
    predictions.
- Responsibility (separation of concerns): Only generate predictions.
    No training, no metrics, no file saving.
- Pipeline contract (inputs and outputs): Input is any DataFrame with the same
    feature columns used during training. Output is a single-column DataFrame.

TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

# Standard Library Imports
import logging

# Third-party Imports
import pandas as pd

logger = logging.getLogger(__name__)


def run_dtrees_inference(model, X_infer):
    """
    Inputs:
    - model: Fitted sklearn Pipeline from dtrees_train.train_model.
    - X_infer: Feature matrix for new patients (pandas DataFrame, no target
    column).
    Outputs:
    - predictions_df: DataFrame with one column "prediction", same index as
    X_infer.
    Why this contract matters for reliable ML delivery:
    - Preserving the input index means predictions can be joined back to the
      original patient records without ambiguity.
    """
    logger.info("Running inference on %d rows...", len(X_infer))

    try:
        predictions = model.predict(X_infer)
    except Exception as exc:
        raise RuntimeError(
            f"[dtrees_infer.run_inference] Inference failed: {exc}"
        ) from exc

    predictions_df = pd.DataFrame(
        {"prediction": predictions},
        index=X_infer.index,
    )

    logger.info("Inference complete.")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add postprocessing here if needed.
    # Why: In production you may want probabilities alongside the label,
    #      or a human-readable output instead of "Presence"/"Absence".
    # Examples:
    # 1. Add probability column:
    #    proba = model.predict_proba(X_infer)[:, 1]
    #    predictions_df["probability"] = proba
    # 2. Map labels to plain English for a clinical report.
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to
    # proceed!")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return predictions_df
