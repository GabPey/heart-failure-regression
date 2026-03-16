from __future__ import annotations

from typing import Any

import pandas as pd
import statsmodels.api as sm


def _fit_ols(X: pd.DataFrame, y: pd.Series):
    """
    Fit an OLS model with an intercept.
    """
    X_model = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_model).fit()
    return model


def _validate_inputs(
    df: pd.DataFrame,
    target: str,
    candidate_features: list[str],
) -> None:
    """
    Validate target and candidate feature names.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    missing = [col for col in candidate_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing candidate features in dataframe: {missing}")

    if target in candidate_features:
        raise ValueError("Target variable must not be included in candidate_features.")


def forward_selection_aic(
    df: pd.DataFrame,
    target: str,
    candidate_features: list[str],
) -> dict[str, Any]:
    """
    Perform forward stepwise selection using AIC.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing target and candidate predictors.
    target : str
        Name of the target variable.
    candidate_features : list[str]
        Candidate predictor names.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - selected_features
        - history_table
        - model
        - aic
        - bic
        - r_squared
        - adj_r_squared
    """
    _validate_inputs(df=df, target=target, candidate_features=candidate_features)

    X_all = df[candidate_features].copy()
    y = df[target].copy()

    remaining = list(candidate_features)
    selected: list[str] = []
    current_aic = float("inf")
    history_rows: list[dict[str, Any]] = []

    step = 0

    while remaining:
        scores = []

        for candidate in remaining:
            features = selected + [candidate]
            model = _fit_ols(X_all[features], y)
            scores.append((model.aic, candidate, features, model))

        scores.sort(key=lambda x: x[0])
        best_aic, best_feature, best_features, best_model = scores[0]

        if best_aic < current_aic:
            step += 1
            selected = best_features
            remaining.remove(best_feature)
            current_aic = best_aic

            history_rows.append(
                {
                    "step": step,
                    "action": "add",
                    "feature": best_feature,
                    "n_features": len(selected),
                    "aic": best_model.aic,
                    "bic": best_model.bic,
                    "r_squared": best_model.rsquared,
                    "adj_r_squared": best_model.rsquared_adj,
                }
            )
        else:
            break

    final_model = _fit_ols(X_all[selected], y)

    return {
        "selected_features": selected,
        "history_table": pd.DataFrame(history_rows),
        "model": final_model,
        "aic": final_model.aic,
        "bic": final_model.bic,
        "r_squared": final_model.rsquared,
        "adj_r_squared": final_model.rsquared_adj,
    }


def backward_selection_aic(
    df: pd.DataFrame,
    target: str,
    candidate_features: list[str],
) -> dict[str, Any]:
    """
    Perform backward stepwise selection using AIC.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing target and candidate predictors.
    target : str
        Name of the target variable.
    candidate_features : list[str]
        Candidate predictor names.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - selected_features
        - history_table
        - model
        - aic
        - bic
        - r_squared
        - adj_r_squared
    """
    _validate_inputs(df=df, target=target, candidate_features=candidate_features)

    X_all = df[candidate_features].copy()
    y = df[target].copy()

    selected = list(candidate_features)
    current_model = _fit_ols(X_all[selected], y)
    current_aic = current_model.aic

    history_rows: list[dict[str, Any]] = []
    step = 0

    while len(selected) > 1:
        scores = []

        for candidate in selected:
            features = [f for f in selected if f != candidate]
            model = _fit_ols(X_all[features], y)
            scores.append((model.aic, candidate, features, model))

        scores.sort(key=lambda x: x[0])
        best_aic, removed_feature, best_features, best_model = scores[0]

        if best_aic < current_aic:
            step += 1
            selected = best_features
            current_aic = best_aic

            history_rows.append(
                {
                    "step": step,
                    "action": "remove",
                    "feature": removed_feature,
                    "n_features": len(selected),
                    "aic": best_model.aic,
                    "bic": best_model.bic,
                    "r_squared": best_model.rsquared,
                    "adj_r_squared": best_model.rsquared_adj,
                }
            )
        else:
            break

    final_model = _fit_ols(X_all[selected], y)

    return {
        "selected_features": selected,
        "history_table": pd.DataFrame(history_rows),
        "model": final_model,
        "aic": final_model.aic,
        "bic": final_model.bic,
        "r_squared": final_model.rsquared,
        "adj_r_squared": final_model.rsquared_adj,
    }


def build_stepwise_comparison_table(
    forward_results: dict[str, Any],
    backward_results: dict[str, Any],
) -> pd.DataFrame:
    """
    Build a compact comparison table for forward vs backward selection.
    """
    return pd.DataFrame(
        [
            {
                "method": "forward_aic",
                "n_features": len(forward_results["selected_features"]),
                "selected_features": ", ".join(forward_results["selected_features"]),
                "aic": forward_results["aic"],
                "bic": forward_results["bic"],
                "r_squared": forward_results["r_squared"],
                "adj_r_squared": forward_results["adj_r_squared"],
            },
            {
                "method": "backward_aic",
                "n_features": len(backward_results["selected_features"]),
                "selected_features": ", ".join(backward_results["selected_features"]),
                "aic": backward_results["aic"],
                "bic": backward_results["bic"],
                "r_squared": backward_results["r_squared"],
                "adj_r_squared": backward_results["adj_r_squared"],
            },
        ]
    )


def run_stepwise_selection_processed(
    df: pd.DataFrame,
    target: str = "ejection_fraction_centered",
    candidate_features: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run both forward and backward AIC-based selection for the processed dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe.
    target : str, default="ejection_fraction_centered"
        Target variable.
    candidate_features : list[str] | None
        Candidate predictors. If None, a default processed-feature set is used.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - forward
        - backward
        - comparison_table
    """
    if candidate_features is None:
        candidate_features = [
            "age_centered",
            "creatinine_log",
            "ejection_creatinine_interaction",
            "sodium_creatinine_interaction",
            "age_diabetes_interaction",
            "serum_sodium",
            "time",
        ]

    forward_results = forward_selection_aic(
        df=df,
        target=target,
        candidate_features=candidate_features,
    )

    backward_results = backward_selection_aic(
        df=df,
        target=target,
        candidate_features=candidate_features,
    )

    comparison_table = build_stepwise_comparison_table(
        forward_results=forward_results,
        backward_results=backward_results,
    )

    return {
        "forward": forward_results,
        "backward": backward_results,
        "comparison_table": comparison_table,
    }
