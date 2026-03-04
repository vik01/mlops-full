"""
Module: Decision Tree Evaluation
----------------------------------
Role: Compute and display performance metrics for the trained Decision Tree.
Input: Fitted Pipeline + X_test, y_test + problem_type (str).
Output: Single float (F1 score) returned to main.py. Full metrics printed to console.

Educational Goal:
- Why this module exists in an MLOps system:
    Separating evaluation from training means you can re-evaluate a saved
    model on new data without retraining.
- Responsibility (separation of concerns): Only compute and report metrics.
    No training, no saving, no plotting.
- Pipeline contract (inputs and outputs): Takes fitted model and test data,
    returns a single float so main.py can log it consistently.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(cm, y_true, y_pred):
    """
    Inputs:
    - cm: Confusion matrix (2x2 numpy array, from sklearn confusion_matrix).
    - y_true: True labels (pandas Series).
    - y_pred: Predicted labels (numpy array).
    Outputs:
    - Dictionary with Accuracy, Precision, Recall, Specificity, F1-score,
      False Positive Rate, TP, TN, FP, FN.
    Why this contract matters for reliable ML delivery:
    - Centralising metric computation means every model on the team reports
      the same numbers in the same format for fair comparison.
    """
    # cm layout: [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="Presence")
    recall = recall_score(y_true, y_pred, pos_label="Presence")
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = f1_score(y_true, y_pred, pos_label="Presence")
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-score": f1,
        "False Positive Rate": fpr,
    }


def evaluate_dtrees_model(model, X_test, y_test, problem_type):
    """
    Inputs:
    - model: Fitted sklearn Pipeline from dtrees_train.train_model.
    - X_test: Held-out feature matrix (pandas DataFrame).
    - y_test: Held-out true labels (pandas Series).
    - problem_type: "classification" or "regression" (str).
    Outputs:
    - metric_value: Single float — F1 score (weighted) for classification.
    Why this contract matters for reliable ML delivery:
    - Returning one float lets main.py log and compare all three models
      (Decision Tree, Logistic Regression, K-Means) on the same scale.
    """
    print("[dtrees_eval.evaluate_model] Running predictions on test set...")  # TODO: replace with logging later

    try:
        y_pred = model.predict(X_test)
    except Exception as exc:
        raise RuntimeError(
            f"[dtrees_eval.evaluate_model] Prediction failed: {exc}"
        ) from exc

    # Full classification report printed to console for visibility
    print("\n[dtrees_eval.evaluate_model] Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=["Absence", "Presence"])
    metrics = calculate_metrics(cm, y_test, y_pred)

    print("[dtrees_eval.evaluate_model] Metrics Summary:")
    for key, val in metrics.items():
        print(f"  {key:<25} {val}")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add additional metrics or plots here if needed.
    # Why: The baseline returns F1 only. You may want ROC-AUC or
    #      a confusion matrix heatmap saved to reports/.
    # Examples:
    # 1. from sklearn.metrics import roc_auc_score
    #    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    # 2. Save a confusion matrix plot to reports/dt_confusion_matrix.png
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # Return single float back to main.py
    metric_value = float(metrics["F1-score"])
    print(f"\n[dtrees_eval.evaluate_model] F1 Score: {metric_value:.4f}")  # TODO: replace with logging later

    return metric_value
