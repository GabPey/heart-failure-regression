"""
src/models/regression_diagnostics.py

Diagnostics for two OLS regression models on the processed dataset.

This module checks the three main issues:
1. Multicollinearity (VIF)
2. Heteroscedasticity (Breusch-Pagan)
3. Influential observations (Cook's distance)

Models supported
----------------
- parsimonious model
- stepwise model

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

Typical notebook usage
----------------------
from src.models.regression_diagnostics import run_regression_diagnostics

results = run_regression_diagnostics(df)

display(results["parsimonious"]["vif_table"])
display(results["stepwise"]["vif_table"])

display(results["heteroscedasticity_table"])
display(results["influential_summary_table"])

print(results["parsimonious"]["model"].summary())
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
    Fit OLS with intercept.
    """
    X_train_const = _add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()
    return model


def compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIF table for a set of predictors.
    """
    X_const = _add_constant(X)

    vif_rows = []
    for i, col in enumerate(X_const.columns):
        vif_rows.append(
            {
                "feature": col,
                "vif": variance_inflation_factor(X_const.values, i),
            }
        )

    vif_table = pd.DataFrame(vif_rows)

    # Usually intercept is not interpreted in VIF discussions
    return vif_table[vif_table["feature"] != "const"].reset_index(drop=True)


def breusch_pagan_test(model) -> dict[str, float]:
    """
    Run Breusch-Pagan test for heteroscedasticity.
    """
    bp_stat, bp_pvalue, f_stat, f_pvalue = het_breuschpagan(
        model.resid,
        model.model.exog,
    )

    return {
        "bp_stat": bp_stat,
        "bp_pvalue": bp_pvalue,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
    }


def cooks_distance_table(model) -> pd.DataFrame:
    """
    Compute Cook's distance for each observation.
    """
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    table = pd.DataFrame(
        {
            "observation": range(len(cooks_d)),
            "cooks_distance": cooks_d,
        }
    ).sort_values("cooks_distance", ascending=False)

    return table.reset_index(drop=True)


def influential_summary(model) -> dict[str, float]:
    """
    Summarize influential observations using Cook's distance.
    """
    cooks_df = cooks_distance_table(model)
    n = len(cooks_df)
    threshold = 4 / n

    n_influential = int((cooks_df["cooks_distance"] > threshold).sum())
    max_cooks = float(cooks_df["cooks_distance"].max())

    return {
        "n_observations": n,
        "cooks_threshold_4_over_n": threshold,
        "n_influential_points": n_influential,
        "max_cooks_distance": max_cooks,
    }


def plot_cooks_distance(
    model,
    title: str,
) -> None:
    """
    Plot Cook's distance with reference threshold.
    """
    cooks_df = cooks_distance_table(model)
    n = len(cooks_df)
    threshold = 4 / n

    plt.figure(figsize=(8, 4.5))
    plt.stem(
        cooks_df["observation"],
        cooks_df["cooks_distance"],
        basefmt=" ",
    )
    plt.axhline(threshold, linestyle="--")
    plt.xlabel("Observation")
    plt.ylabel("Cook's distance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_top_cooks_distance(
    model,
    title: str,
    top_n: int = 10,
) -> None:
    """
    Plot the top-N most influential observations.
    """
    cooks_df = cooks_distance_table(model).head(top_n)

    plt.figure(figsize=(8, 4.5))
    plt.bar(cooks_df["observation"].astype(str), cooks_df["cooks_distance"])
    plt.xlabel("Observation")
    plt.ylabel("Cook's distance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _diagnose_one_model(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train one model and compute main diagnostics on the training set.
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

    vif_table = compute_vif_table(X_train)
    bp_results = breusch_pagan_test(model)
    cooks_df = cooks_distance_table(model)
    cooks_summary = influential_summary(model)

    return {
        "model_name": model_name,
        "features": features,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "vif_table": vif_table,
        "bp_results": bp_results,
        "cooks_distance_table": cooks_df,
        "cooks_summary": cooks_summary,
    }


def build_heteroscedasticity_table(
    parsimonious_results: dict[str, Any],
    stepwise_results: dict[str, Any],
) -> pd.DataFrame:
    """
    Build a comparison table for Breusch-Pagan results.
    """
    rows = []

    for results in [parsimonious_results, stepwise_results]:
        rows.append(
            {
                "model": results["model_name"],
                "bp_stat": results["bp_results"]["bp_stat"],
                "bp_pvalue": results["bp_results"]["bp_pvalue"],
                "f_stat": results["bp_results"]["f_stat"],
                "f_pvalue": results["bp_results"]["f_pvalue"],
            }
        )

    return pd.DataFrame(rows)


def build_influential_summary_table(
    parsimonious_results: dict[str, Any],
    stepwise_results: dict[str, Any],
) -> pd.DataFrame:
    """
    Build a comparison table for Cook's distance summaries.
    """
    rows = []

    for results in [parsimonious_results, stepwise_results]:
        rows.append(
            {
                "model": results["model_name"],
                "n_observations": results["cooks_summary"]["n_observations"],
                "cooks_threshold_4_over_n": results["cooks_summary"]["cooks_threshold_4_over_n"],
                "n_influential_points": results["cooks_summary"]["n_influential_points"],
                "max_cooks_distance": results["cooks_summary"]["max_cooks_distance"],
            }
        )

    return pd.DataFrame(rows)


def plot_cooks_distance_comparison(
    parsimonious_results: dict[str, Any],
    stepwise_results: dict[str, Any],
) -> None:
    """
    Side-by-side Cook's distance plots for both models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, title in [
        (axes[0], parsimonious_results, "Parsimonious model: Cook's distance"),
        (axes[1], stepwise_results, "Stepwise model: Cook's distance"),
    ]:
        cooks_df = results["cooks_distance_table"]
        threshold = results["cooks_summary"]["cooks_threshold_4_over_n"]

        markerline, stemlines, baseline = ax.stem(
            cooks_df["observation"],
            cooks_df["cooks_distance"],
            basefmt=" ",
        )
        ax.axhline(threshold, linestyle="--")
        ax.set_xlabel("Observation")
        ax.set_ylabel("Cook's distance")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_top_cooks_distance_comparison(
    parsimonious_results: dict[str, Any],
    stepwise_results: dict[str, Any],
    top_n: int = 10,
) -> None:
    """
    Side-by-side barplots of the top influential observations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, title in [
        (axes[0], parsimonious_results, "Parsimonious model: top Cook's distance"),
        (axes[1], stepwise_results, "Stepwise model: top Cook's distance"),
    ]:
        cooks_df = results["cooks_distance_table"].head(top_n)
        ax.bar(cooks_df["observation"].astype(str), cooks_df["cooks_distance"])
        ax.set_xlabel("Observation")
        ax.set_ylabel("Cook's distance")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def run_regression_diagnostics(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    parsimonious_features: list[str] | None = None,
    stepwise_features: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    make_plots: bool = True,
    top_n_cooks: int = 10,
) -> dict[str, Any]:
    """
    Run the main diagnostics for both regression models.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - parsimonious
        - stepwise
        - heteroscedasticity_table
        - influential_summary_table
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

    parsimonious_results = _diagnose_one_model(
        df=df,
        target=target,
        features=parsimonious_features,
        model_name="parsimonious",
        test_size=test_size,
        random_state=random_state,
    )

    stepwise_results = _diagnose_one_model(
        df=df,
        target=target,
        features=stepwise_features,
        model_name="stepwise",
        test_size=test_size,
        random_state=random_state,
    )

    heteroscedasticity_table = build_heteroscedasticity_table(
        parsimonious_results=parsimonious_results,
        stepwise_results=stepwise_results,
    )

    influential_summary_table = build_influential_summary_table(
        parsimonious_results=parsimonious_results,
        stepwise_results=stepwise_results,
    )

    if make_plots:
        plot_cooks_distance_comparison(
            parsimonious_results=parsimonious_results,
            stepwise_results=stepwise_results,
        )
        plot_top_cooks_distance_comparison(
            parsimonious_results=parsimonious_results,
            stepwise_results=stepwise_results,
            top_n=top_n_cooks,
        )

    return {
        "parsimonious": parsimonious_results,
        "stepwise": stepwise_results,
        "heteroscedasticity_table": heteroscedasticity_table,
        "influential_summary_table": influential_summary_table,
    }
