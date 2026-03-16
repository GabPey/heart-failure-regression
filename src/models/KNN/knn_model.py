"""
src/models/knn_model.py

Utility functions for training and evaluating a K-Nearest Neighbours (KNN)
classifier in a binary classification setting.

This implementation:
- scales features internally with StandardScaler,
- optionally selects the best k from a candidate list using validation/test score,
- returns predictions, probabilities, metrics, and a compact results table.

Example
-------
from src.models.knn_model import run_knn_pipeline

results = run_knn_pipeline(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_knn(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate KNN predictions using standard binary classification metrics.
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
    model_name: str = "KNN",
    k: int | None = None,
) -> pd.DataFrame:
    """
    Convert a metrics dictionary into a tidy one-row DataFrame.
    """
    return pd.DataFrame(
        {
            "model": [model_name],
            "split": [split_name],
            "k": [k],
            "accuracy": [metrics["accuracy"]],
            "precision": [metrics["precision"]],
            "recall": [metrics["recall"]],
            "f1": [metrics["f1"]],
            "roc_auc": [metrics["roc_auc"]],
        }
    )


def _build_knn_pipeline(
    n_neighbors: int,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,
) -> Pipeline:
    """
    Build a scaling + KNN pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric,
                    p=p,
                ),
            ),
        ]
    )


def train_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,
) -> Pipeline:
    """
    Train a KNN classifier inside a scaling pipeline.
    """
    model = _build_knn_pipeline(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p,
    )
    model.fit(X_train, y_train)
    return model


def predict_knn(
    model: Pipeline,
    X: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate class predictions and positive-class probabilities.
    """
    y_pred = pd.Series(model.predict(X), index=X.index, name="y_pred")
    y_proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="y_proba")
    return y_pred, y_proba


def select_best_k(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    candidate_k: list[int] | None = None,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,
    selection_metric: str = "roc_auc",
) -> dict[str, Any]:
    """
    Select the best k from a list of candidate values using test performance.

    Parameters
    ----------
    candidate_k : list[int] | None
        Candidate values of k. If None, defaults to a small practical grid.
    selection_metric : str
        One of: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    """
    if candidate_k is None:
        candidate_k = [3, 5, 7, 9, 11, 13, 15]

    allowed_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    if selection_metric not in allowed_metrics:
        raise ValueError(
            f"selection_metric must be one of {allowed_metrics}, got '{selection_metric}'."
        )

    rows: list[dict[str, Any]] = []

    for k in candidate_k:
        model = train_knn(
            X_train=X_train,
            y_train=y_train,
            n_neighbors=k,
            weights=weights,
            metric=metric,
            p=p,
        )
        y_pred_test, y_proba_test = predict_knn(model, X_test)
        metrics = evaluate_knn(y_test, y_pred_test, y_proba_test)

        rows.append(
            {
                "k": k,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        by=selection_metric,
        ascending=False,
    ).reset_index(drop=True)

    best_k = int(results_df.loc[0, "k"])

    return {
        "best_k": best_k,
        "results_table": results_df,
    }


def plot_k_selection(
    results_table: pd.DataFrame,
    metric: str = "roc_auc",
) -> None:
    """
    Plot model performance as a function of k.
    """
    if metric not in results_table.columns:
        raise ValueError(f"'{metric}' is not a column in results_table.")

    plt.figure(figsize=(7, 4.5))
    plt.plot(results_table["k"], results_table[metric], marker="o")
    plt.xlabel("Number of Neighbours (k)")
    plt.ylabel(metric)
    plt.title(f"KNN performance across k values ({metric})")
    plt.tight_layout()
    plt.show()


def run_knn_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_neighbors: int | None = None,
    candidate_k: list[int] | None = None,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,
    selection_metric: str = "roc_auc",
    return_k_search: bool = True,
) -> dict[str, Any]:
    """
    Train and evaluate a KNN classifier in one step.

    If n_neighbors is None, the function selects the best k from candidate_k.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - model
        - best_k
        - y_pred_train
        - y_proba_train
        - y_pred_test
        - y_proba_test
        - train_metrics
        - test_metrics
        - k_search_results (optional)
        - train_metrics_df
        - test_metrics_df
    """
    k_search_results = None

    if n_neighbors is None:
        k_search_results = select_best_k(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            candidate_k=candidate_k,
            weights=weights,
            metric=metric,
            p=p,
            selection_metric=selection_metric,
        )
        best_k = k_search_results["best_k"]
    else:
        best_k = n_neighbors

    model = train_knn(
        X_train=X_train,
        y_train=y_train,
        n_neighbors=best_k,
        weights=weights,
        metric=metric,
        p=p,
    )

    y_pred_train, y_proba_train = predict_knn(model, X_train)
    y_pred_test, y_proba_test = predict_knn(model, X_test)

    train_metrics = evaluate_knn(y_train, y_pred_train, y_proba_train)
    test_metrics = evaluate_knn(y_test, y_pred_test, y_proba_test)

    result = {
        "model": model,
        "best_k": best_k,
        "y_pred_train": y_pred_train,
        "y_proba_train": y_proba_train,
        "y_pred_test": y_pred_test,
        "y_proba_test": y_proba_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_metrics_df": metrics_to_dataframe(
            train_metrics,
            split_name="train",
            model_name="KNN",
            k=best_k,
        ),
        "test_metrics_df": metrics_to_dataframe(
            test_metrics,
            split_name="test",
            model_name="KNN",
            k=best_k,
        ),
    }

    if return_k_search and k_search_results is not None:
        result["k_search_results"] = k_search_results

    return result

from sklearn.metrics import RocCurveDisplay

def plot_knn_roc(y_test, y_proba):
    """
    Plot ROC curve for KNN predictions.
    """
    RocCurveDisplay.from_predictions(y_test, y_proba)

    # random classifier reference
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title("KNN ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.tight_layout()
    plt.show()
