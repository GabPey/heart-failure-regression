"""
src/models/regression_comparison.py

Train, evaluate, and visualize two OLS regression models on the processed dataset:

1. Parsimonious model
2. Stepwise-selected model

This module is designed to be called from a notebook with minimal code.

Default target
--------------
ejection_fraction_centered

Default parsimonious features
-----------------------------
- ejection_creatinine_interaction
- creatinine_log
- sodium_creatinine_interaction

Default stepwise features
-------------------------
- age_centered
- creatinine_log
- ejection_creatinine_interaction
- sodium_creatinine_interaction
- serum_sodium

Main notebook usage
-------------------
from src.models.regression_comparison import run_regression_model_comparison

results = run_regression_model_comparison(df)

display(results["metrics_table"])
print(results["parsimonious"]["train_summary"])
print(results["stepwise"]["train_summary"])
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


DEFAULT_TARGET = "ejection_fraction_centered"

DEFAULT_PARSIMONIOUS_FEATURES = [
    "ejection_creatinine_interaction",
    "creatinine_log",
    "sodium_creatinine_interaction",
]

DEFAULT_STEPWISE_FEATURES = [
    "age_centered",
    "creatinine_log",
    "ejection_creatinine_interaction",
    "sodium_creatinine_interaction",
    "serum_sodium",
]


def _validate_inputs(
    df: pd.DataFrame,
    target: str,
    parsimonious_features: list[str],
    stepwise_features: list[str],
) -> None:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    missing_pars = [c for c in parsimonious_features if c not in df.columns]
    missing_step = [c for c in stepwise_features if c not in df.columns]

    if missing_pars:
        raise ValueError(f"Missing parsimonious features: {missing_pars}")

    if missing_step:
        raise ValueError(f"Missing stepwise features: {missing_step}")


def _add_constant(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X, has_constant="add")


def train_ols_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Fit an OLS model with intercept using statsmodels.
    """
    X_train_const = _add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()
    return model


def predict_ols_model(
    model,
    X: pd.DataFrame,
) -> pd.Series:
    """
    Generate predictions from a fitted statsmodels OLS model.
    """
    X_const = _add_constant(X)
    y_pred = model.predict(X_const)
    return pd.Series(y_pred, index=X.index, name="y_pred")


def regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """
    Compute standard regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": rmse,
    }


def _one_model_results(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train/test workflow for a single regression model.
    """
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = train_ols_model(X_train, y_train)

    y_pred_train = predict_ols_model(model, X_train)
    y_pred_test = predict_ols_model(model, X_test)

    train_metrics = regression_metrics(y_train, y_pred_train)
    test_metrics = regression_metrics(y_test, y_pred_test)

    return {
        "model_name": model_name,
        "features": features,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_summary": model.summary(),
    }


def build_metrics_table(
    parsimonious_results: dict[str, Any],
    stepwise_results: dict[str, Any],
) -> pd.DataFrame:
    """
    Build a tidy comparison table for train and test metrics.
    """
    rows = []

    for results in [parsimonious_results, stepwise_results]:
        for split_name, metrics in [
            ("train", results["train_metrics"]),
            ("test", results["test_metrics"]),
        ]:
            rows.append(
                {
                    "model": results["model_name"],
                    "split": split_name,
                    "n_features": len(results["features"]),
                    "features": ", ".join(results["features"]),
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "mse": metrics["mse"],
                    "rmse": metrics["rmse"],
                }
            )

    return pd.DataFrame(rows)


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
) -> None:
    """
    Scatter plot of actual vs predicted values.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_residuals_vs_fitted(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
) -> None:
    """
    Residuals vs fitted values plot.
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")

    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_test_predictions_comparison(
    parsimonious_results: dict[str, Any],
    stepwise_results: dict[str, Any],
) -> None:
    """
    Side-by-side actual vs predicted plots for test predictions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parsimonious
    y_true = parsimonious_results["y_test"]
    y_pred = parsimonious_results["y_pred_test"]
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    axes[0].scatter(y_true, y_pred, alpha=0.7)
    axes[0].plot([min_val, max_val], [min_val, max_val], linestyle="--")
    axes[0].set_title("Parsimonious model: actual vs predicted")
    axes[0].set_xlabel("Actual values")
    axes[0].set_ylabel("Predicted values")

    # Stepwise
    y_true = stepwise_results["y_test"]
    y_pred = stepwise_results["y_pred_test"]
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    axes[1].scatter(y_true, y_pred, alpha=0.7)
    axes[1].plot([min_val, max_val], [min_val, max_val], linestyle="--")
    axes[1].set_title("Stepwise model: actual vs predicted")
    axes[1].set_xlabel("Actual values")
    axes[1].set_ylabel("Predicted values")

    plt.tight_layout()
    plt.show()


def plot_test_residuals_comparison(
    parsimonious_results: dict[str, Any],
    stepwise_results: dict[str, Any],
) -> None:
    """
    Side-by-side residuals vs fitted plots for test predictions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parsimonious
    y_true = parsimonious_results["y_test"]
    y_pred = parsimonious_results["y_pred_test"]
    residuals = y_true - y_pred

    axes[0].scatter(y_pred, residuals, alpha=0.7)
    axes[0].axhline(0, linestyle="--")
    axes[0].set_title("Parsimonious model: residuals vs fitted")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")

    # Stepwise
    y_true = stepwise_results["y_test"]
    y_pred = stepwise_results["y_pred_test"]
    residuals = y_true - y_pred

    axes[1].scatter(y_pred, residuals, alpha=0.7)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_title("Stepwise model: residuals vs fitted")
    axes[1].set_xlabel("Fitted values")
    axes[1].set_ylabel("Residuals")

    plt.tight_layout()
    plt.show()


def plot_r2_comparison(
    metrics_table: pd.DataFrame,
) -> None:
    """
    Plot test R² for both models.
    """
    test_df = metrics_table[metrics_table["split"] == "test"].copy()

    plt.figure(figsize=(6, 4.5))
    plt.bar(test_df["model"], test_df["r2"])
    plt.ylabel("Test $R^2$")
    plt.title("Test R² comparison")
    plt.tight_layout()
    plt.show()


def run_regression_model_comparison(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    parsimonious_features: list[str] | None = None,
    stepwise_features: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    make_plots: bool = True,
) -> dict[str, Any]:
    """
    Train, evaluate, and optionally plot results for the parsimonious and stepwise models.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - parsimonious
        - stepwise
        - metrics_table
    """
    if parsimonious_features is None:
        parsimonious_features = DEFAULT_PARSIMONIOUS_FEATURES

    if stepwise_features is None:
        stepwise_features = DEFAULT_STEPWISE_FEATURES

    _validate_inputs(
        df=df,
        target=target,
        parsimonious_features=parsimonious_features,
        stepwise_features=stepwise_features,
    )

    parsimonious_results = _one_model_results(
        df=df,
        target=target,
        features=parsimonious_features,
        model_name="parsimonious",
        test_size=test_size,
        random_state=random_state,
    )

    stepwise_results = _one_model_results(
        df=df,
        target=target,
        features=stepwise_features,
        model_name="stepwise",
        test_size=test_size,
        random_state=random_state,
    )

    metrics_table = build_metrics_table(
        parsimonious_results=parsimonious_results,
        stepwise_results=stepwise_results,
    )

    if make_plots:
        plot_test_predictions_comparison(
            parsimonious_results=parsimonious_results,
            stepwise_results=stepwise_results,
        )
        plot_test_residuals_comparison(
            parsimonious_results=parsimonious_results,
            stepwise_results=stepwise_results,
        )
        plot_r2_comparison(metrics_table)

    return {
        "parsimonious": parsimonious_results,
        "stepwise": stepwise_results,
        "metrics_table": metrics_table,
    }
