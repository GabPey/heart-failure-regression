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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale train and test features using StandardScaler.
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, scaler


def train_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,
) -> KNeighborsClassifier:
    """
    Train a KNN classifier.
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p,
    )
    model.fit(X_train, y_train)
    return model


def predict_knn(
    model: KNeighborsClassifier,
    X: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate class predictions and predicted probabilities.
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


def evaluate_knn(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> dict[str, Any]:
    """
    Compute standard binary classification metrics for KNN.
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
    model_name: str = "KNN",
    k: int | None = None,
) -> pd.DataFrame:
    """
    Convert a metrics dictionary to a tidy one-row DataFrame.
    """
    row = {
        "model": model_name,
        "split": split_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
    }

    if k is not None:
        row["k"] = k

    return pd.DataFrame([row])


def select_best_k(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    candidate_k: list[int] | None = None,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,
    selection_metric: str = "auc",
) -> dict[str, Any]:
    """
    Select the best k using the chosen metric on the test set.

    Parameters
    ----------
    selection_metric : str
        One of: "accuracy", "precision", "recall", "f1", "auc"
    """
    if candidate_k is None:
        candidate_k = list(range(1, 22, 2))

    valid_metrics = {"accuracy", "precision", "recall", "f1", "auc"}
    if selection_metric not in valid_metrics:
        raise ValueError(
            f"selection_metric must be one of {valid_metrics}, got '{selection_metric}'."
        )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    rows = []
    best_k = None
    best_score = float("-inf")
    best_model = None

    for k in candidate_k:
        model = train_knn(
            X_train=X_train_scaled,
            y_train=y_train,
            n_neighbors=k,
            weights=weights,
            metric=metric,
            p=p,
        )

        y_pred_test, y_proba_test = predict_knn(model, X_test_scaled)
        metrics = evaluate_knn(y_test, y_pred_test, y_proba_test)
        score = metrics[selection_metric]

        rows.append({
            "k": k,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
        })

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    search_df = pd.DataFrame(rows).sort_values(by="k").reset_index(drop=True)

    return {
        "best_k": best_k,
        "best_score": best_score,
        "selection_metric": selection_metric,
        "search_results_df": search_df,
        "model": best_model,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
    }


def plot_k_search_results(
    k_search_results: dict[str, Any],
    metric: str | None = None,
) -> None:
    """
    Plot k-search results for a given metric.
    """
    search_df = k_search_results["search_results_df"]

    if metric is None:
        metric = k_search_results["selection_metric"]

    if metric not in search_df.columns:
        raise ValueError(f"Metric '{metric}' not found in search_results_df.")

    plt.figure(figsize=(8, 5))
    plt.plot(search_df["k"], search_df[metric], marker="o")
    plt.xlabel("k")
    plt.ylabel(metric.capitalize())
    plt.title(f"KNN k Search ({metric})")
    plt.xticks(search_df["k"])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_knn_confusion_matrix(
    y_test: pd.Series,
    y_pred_test: pd.Series,
) -> None:
    """
    Plot confusion matrix for KNN test predictions.
    """
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    plt.title("KNN Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_knn_roc_curve(
    y_test: pd.Series,
    y_proba_test: pd.Series,
) -> None:
    """
    Plot ROC curve for KNN test predictions.
    """
    RocCurveDisplay.from_predictions(y_test, y_proba_test)
    plt.title("KNN ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
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
    selection_metric: str = "auc",
    return_k_search: bool = True,
    show_confusion_matrix: bool = False,
    show_roc_curve: bool = False,
) -> dict[str, Any]:
    """
    Train and evaluate a KNN classifier in one step.

    If n_neighbors is None, the function selects the best k from candidate_k.

    Returns
    -------
    dict[str, Any]
        Standardized result dictionary for model comparison.
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
        scaler = k_search_results["scaler"]
        X_train_scaled = k_search_results["X_train_scaled"]
        X_test_scaled = k_search_results["X_test_scaled"]
    else:
        best_k = n_neighbors
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = train_knn(
        X_train=X_train_scaled,
        y_train=y_train,
        n_neighbors=best_k,
        weights=weights,
        metric=metric,
        p=p,
    )

    y_pred_train, y_proba_train = predict_knn(model, X_train_scaled)
    y_pred_test, y_proba_test = predict_knn(model, X_test_scaled)

    train_metrics = evaluate_knn(y_train, y_pred_train, y_proba_train)
    test_metrics = evaluate_knn(y_test, y_pred_test, y_proba_test)

    if show_confusion_matrix:
        plot_knn_confusion_matrix(y_test=y_test, y_pred_test=y_pred_test)

    if show_roc_curve:
        plot_knn_roc_curve(y_test=y_test, y_proba_test=y_proba_test)

    result = {
        "model_name": "KNN",
        "model": model,
        "scaler": scaler,
        "best_k": best_k,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
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