"""
Educational Goal:
- Why this module exists in an MLOps system: Evaluation provides a
  consistent, testable way to compare model iterations.
- Responsibility (separation of concerns): This module only computes
  metrics; it does not train models or save artifacts.
- Pipeline contract (inputs and outputs): Input is a fitted model Pipeline
  and held-out test set; output is a dictionary of performance metrics.

TODO: Replace print statements with standard library logging in a later
session
TODO: Any temporary or hardcoded variable or parameter will be imported
from config.yml in a later session
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)


_INFERENCE_DIR = os.path.join("data", "inference")


def __plot_roc_curve(fpr, tpr, auc_score, save_path):
    """Plot ROC curve with AUC score and save to file."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2,
             label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--",
             label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity/Recall)")
    plt.title("ROC Curve - Test Set")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {save_path}")


def __find_optimal_threshold(fpr, tpr, thresholds):
    """Select optimal threshold using Youden's J statistic."""
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def __calculate_metrics(cm, y_true, y_pred):
    """Calculate classification metrics from confusion matrix."""
    TN, FP, FN, TP = cm.ravel()

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

    return {
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred)),
        "Specificity": float(specificity),
        "F1-score": float(f1_score(y_true, y_pred)),
        "False Positive Rate": float(fpr),
    }


def __plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix heatmap for the test set and save to file."""
    labels = ["Absence", "Presence"]
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Test Set (Optimal Threshold)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def __plot_coefficients(model, save_path):
    """Plot logistic regression coefficients as a horizontal bar chart."""
    estimator = model.named_steps["model"]
    feature_names = model.named_steps["preprocess"].get_feature_names_out()

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": estimator.coef_[0],
    }).sort_values(by="Coefficient", ascending=True)

    plt.figure(figsize=(10, 6))
    colors = ["#E45756" if c < 0 else "#4C78A8"
              for c in coef_df["Coefficient"]]
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    plt.xlabel("Coefficient Value")
    plt.title("Logistic Regression Coefficients")
    plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Coefficients chart saved to {save_path}")


def evaluate_logit_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, prob_type: str
) -> dict:
    """
    Inputs:
    - model: fitted sklearn Pipeline
    - X_test: test features dataframe
    - y_test: test target series
    - prob_type: "regression" or "classification"
    Outputs:
    - dict of performance metrics (optimal threshold, test set)
    """
    print("evaluate_model: evaluating model on held-out test set")

    if prob_type not in ["regression", "classification"]:
        msg = (
            'evaluate_model: problem_type must be '
            '"regression" or "classification"'
        )
        raise ValueError(msg)

    os.makedirs(_INFERENCE_DIR, exist_ok=True)

    # 1. ROC curve and AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    roc_path = os.path.join(_INFERENCE_DIR, "logit_roc.jpg")
    __plot_roc_curve(fpr, tpr, auc_score, roc_path)

    # 2. Optimal threshold via Youden's J
    optimal_threshold = __find_optimal_threshold(fpr, tpr, thresholds)

    # 3. Predictions at optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

    # 4. Confusion matrix (test set only)
    cm = confusion_matrix(y_test, y_pred_optimal)
    cm_path = os.path.join(_INFERENCE_DIR, "logit_optimal_roc.jpg")
    __plot_confusion_matrix(cm, cm_path)

    # 5. Metrics
    metrics = __calculate_metrics(cm, y_test, y_pred_optimal)
    metrics["optimal_threshold"] = optimal_threshold

    # 6. Coefficient bar chart
    coef_path = os.path.join(_INFERENCE_DIR, "logit_coefficients.jpg")
    __plot_coefficients(model, coef_path)

    return metrics
