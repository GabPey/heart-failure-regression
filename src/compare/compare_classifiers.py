from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve


def _validate_result_dict(result: dict[str, Any]) -> None:
    """
    Validate that a model result dictionary follows the standardized structure.
    """
    required_keys = [
        "model_name",
        "y_test",
        "y_pred_test",
        "y_proba_test",
        "test_metrics",
    ]

    missing = [key for key in required_keys if key not in result]
    if missing:
        raise ValueError(
            f"Model result dictionary is missing required keys: {missing}"
        )

    required_metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
    missing_metrics = [
        key for key in required_metric_keys if key not in result["test_metrics"]
    ]
    if missing_metrics:
        raise ValueError(
            f"test_metrics is missing required keys: {missing_metrics}"
        )


def build_comparison_table(
    results_list: list[dict[str, Any]],
    round_digits: int = 4,
    sort_by: str = "auc",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Build a comparative metrics table from standardized model result dictionaries.
    """
    rows = []

    for result in results_list:
        _validate_result_dict(result)

        metrics = result["test_metrics"]

        row = {
            "Model": result["model_name"],
            "Accuracy": round(metrics["accuracy"], round_digits),
            "Precision": round(metrics["precision"], round_digits),
            "Recall": round(metrics["recall"], round_digits),
            "F1": round(metrics["f1"], round_digits),
            "AUC": round(metrics["auc"], round_digits),
        }

        if "best_k" in result:
            row["best_k"] = result["best_k"]

        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    sort_map = {
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "auc": "AUC",
        "best_k": "best_k",
    }

    sort_column = sort_map.get(sort_by.lower(), "AUC")
    if sort_column in comparison_df.columns:
        comparison_df = comparison_df.sort_values(
            by=sort_column,
            ascending=ascending,
        ).reset_index(drop=True)

    return comparison_df


def plot_model_rocs(
    results_list: list[dict[str, Any]],
    title: str = "ROC Curve Comparison",
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """
    Plot ROC curves for multiple standardized model result dictionaries.
    """
    plt.figure(figsize=figsize)

    for result in results_list:
        _validate_result_dict(result)

        model_name = result["model_name"]
        y_test = result["y_test"]
        y_proba = result["y_proba_test"]
        auc_value = result["test_metrics"]["auc"]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {auc_value:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_models(
    results_list: list[dict[str, Any]],
    title: str = "ROC Curve Comparison",
    round_digits: int = 4,
    sort_by: str = "auc",
    ascending: bool = False,
    figsize: tuple[int, int] = (8, 6),
) -> pd.DataFrame:
    """
    Plot ROC curves and return a comparative metrics table.

    Parameters
    ----------
    results_list : list[dict[str, Any]]
        List of standardized model output dictionaries.
    title : str
        Plot title for ROC comparison.
    round_digits : int
        Number of decimal places for the output table.
    sort_by : str
        Metric to sort the comparison table by.
    ascending : bool
        Sort order for the comparison table.
    figsize : tuple[int, int]
        Figure size for the ROC plot.

    Returns
    -------
    pd.DataFrame
        Comparative metrics table.
    """
    plot_model_rocs(
        results_list=results_list,
        title=title,
        figsize=figsize,
    )

    comparison_df = build_comparison_table(
        results_list=results_list,
        round_digits=round_digits,
        sort_by=sort_by,
        ascending=ascending,
    )

    return comparison_df