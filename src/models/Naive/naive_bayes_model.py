"""
src/models/naive_bayes_model.py

Training and evaluation utilities for Gaussian Naive Bayes classification.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)


def train_naive_bayes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GaussianNB:
    """
    Train Gaussian Naive Bayes classifier.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def predict_naive_bayes(
    model: GaussianNB,
    X: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate predictions and probabilities.
    """
    y_pred = pd.Series(model.predict(X), index=X.index, name="y_pred")
    y_proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="y_proba")

    return y_pred, y_proba


def evaluate_naive_bayes(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> dict[str, Any]:
    """
    Compute classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def metrics_to_dataframe(
    metrics: dict[str, Any],
    split_name: str = "test",
    model_name: str = "Naive Bayes",
) -> pd.DataFrame:
    """
    Convert metrics dictionary into a tidy DataFrame row.
    """
    return pd.DataFrame(
        {
            "model": [model_name],
            "split": [split_name],
            "accuracy": [metrics["accuracy"]],
            "precision": [metrics["precision"]],
            "recall": [metrics["recall"]],
            "f1": [metrics["f1"]],
            "roc_auc": [metrics["roc_auc"]],
        }
    )


def plot_nb_roc(
    y_test: pd.Series,
    y_proba: pd.Series,
) -> None:
    """
    Plot ROC curve.
    """
    RocCurveDisplay.from_predictions(y_test, y_proba)

    # random classifier reference
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title("Naive Bayes ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.tight_layout()
    plt.show()


def run_naive_bayes_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Train and evaluate Naive Bayes in one step.
    """

    model = train_naive_bayes(X_train, y_train)

    y_pred_train, y_proba_train = predict_naive_bayes(model, X_train)
    y_pred_test, y_proba_test = predict_naive_bayes(model, X_test)

    train_metrics = evaluate_naive_bayes(y_train, y_pred_train, y_proba_train)
    test_metrics = evaluate_naive_bayes(y_test, y_pred_test, y_proba_test)

    return {
        "model": model,
        "y_pred_train": y_pred_train,
        "y_proba_train": y_proba_train,
        "y_pred_test": y_pred_test,
        "y_proba_test": y_proba_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_metrics_df": metrics_to_dataframe(train_metrics, "train"),
        "test_metrics_df": metrics_to_dataframe(test_metrics, "test"),
    }
