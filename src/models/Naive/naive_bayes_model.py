from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB


def train_naive_bayes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GaussianNB:
    """
    Train a Gaussian Naive Bayes classifier.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def predict_naive_bayes(
    model: GaussianNB,
    X: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate predictions and predicted probabilities.
    """
    y_pred = pd.Series(
        model.predict(X),
        index=X.index,
        name="y_pred",
    )

    y_proba = pd.Series(
        model.predict_proba(X)[:, 1],
        index=X.index,
        name="y_proba",
    )

    return y_pred, y_proba


def evaluate_naive_bayes(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> dict[str, Any]:
    """
    Compute binary classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def metrics_to_dataframe(
    metrics: dict[str, Any],
    split_name: str,
    model_name: str = "Naive Bayes",
) -> pd.DataFrame:
    """
    Convert metrics dictionary into a tidy DataFrame row.
    """
    return pd.DataFrame([{
        "model": model_name,
        "split": split_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
    }])


def plot_nb_confusion_matrix(
    y_test: pd.Series,
    y_pred_test: pd.Series,
) -> None:
    """
    Plot confusion matrix for test predictions.
    """
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    plt.title("Naive Bayes Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_nb_roc_curve(
    y_test: pd.Series,
    y_proba_test: pd.Series,
) -> None:
    """
    Plot ROC curve for test predictions.
    """
    RocCurveDisplay.from_predictions(y_test, y_proba_test)
    plt.title("Naive Bayes ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.tight_layout()
    plt.show()


def run_naive_bayes_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    show_confusion_matrix: bool = False,
    show_roc_curve: bool = False,
) -> dict[str, Any]:
    """
    Train and evaluate Naive Bayes in one step.

    Returns
    -------
    dict[str, Any]
        Standardized result dictionary compatible with
        Logistic, LDA, and KNN pipelines.
    """

    model = train_naive_bayes(X_train, y_train)

    y_pred_train, y_proba_train = predict_naive_bayes(model, X_train)
    y_pred_test, y_proba_test = predict_naive_bayes(model, X_test)

    train_metrics = evaluate_naive_bayes(y_train, y_pred_train, y_proba_train)
    test_metrics = evaluate_naive_bayes(y_test, y_pred_test, y_proba_test)

    if show_confusion_matrix:
        plot_nb_confusion_matrix(y_test, y_pred_test)

    if show_roc_curve:
        plot_nb_roc_curve(y_test, y_proba_test)

    return {
        "model_name": "Naive Bayes",
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_proba_train": y_proba_train,
        "y_pred_test": y_pred_test,
        "y_proba_test": y_proba_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_metrics_df": metrics_to_dataframe(train_metrics, "train"),
        "test_metrics_df": metrics_to_dataframe(test_metrics, "test"),
    }