from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve


def _metrics_dict_to_df(
    metrics: dict[str, Any],
    split_name: str,
    round_digits: int = 4,
) -> pd.DataFrame:
    """
    Convert metrics dict to a one-row DataFrame, excluding non-scalar entries
    such as confusion matrices.
    """
    scalar_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }

    df = pd.DataFrame([scalar_metrics], index=[split_name])
    return df.round(round_digits)


def display_model_results(
    result: dict[str, Any],
    show_train_metrics: bool = True,
    show_confusion_matrix: bool = True,
    show_roc_curve: bool = True,
    round_digits: int = 4,
) -> None:
    """
    Display a standardized model result dictionary in a friendly format.

    Parameters
    ----------
    result : dict
        Standardized model result dictionary produced by any pipeline.
    show_train_metrics : bool
        Whether to display training metrics.
    show_confusion_matrix : bool
        Whether to plot the confusion matrix.
    show_roc_curve : bool
        Whether to plot the ROC curve.
    round_digits : int
        Number of decimals to display in metric tables.
    """
    model_name = result["model_name"]

    print("\n" + "=" * 60)
    print(f"MODEL RESULTS: {model_name}")
    print("=" * 60)

    train_metrics = result["train_metrics"]
    test_metrics = result["test_metrics"]

    train_df = _metrics_dict_to_df(
        train_metrics,
        split_name="train",
        round_digits=round_digits,
    )
    test_df = _metrics_dict_to_df(
        test_metrics,
        split_name="test",
        round_digits=round_digits,
    )

    metrics_df = pd.concat([train_df, test_df]) if show_train_metrics else test_df

    print("\nMetrics:")
    try:
        from IPython.display import display
        display(metrics_df)
    except ImportError:
        print(metrics_df)

    if show_confusion_matrix:
        y_test = result["y_test"]
        y_pred = result["y_pred_test"]

        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title(f"{model_name} Confusion Matrix")
        plt.tight_layout()
        plt.show()

    if show_roc_curve:
        y_test = result["y_test"]
        y_proba = result["y_proba_test"]

        fpr, tpr, _ = roc_curve(y_test, y_proba)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {test_metrics['auc']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    if "best_k" in result:
        print(f"\nBest k: {result['best_k']}")