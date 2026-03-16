import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def plot_logit_vs_feature(
    df: pd.DataFrame,
    features: list[str],
    model,
    X: pd.DataFrame,
    ncols: int = 2,
) -> None:
    """
    Plot feature vs logit(predicted probability) with LOWESS smoothing
    to visually inspect possible non-linearity.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]

    eps = 1e-6
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    logit_pred = np.log(y_pred_proba / (1 - y_pred_proba))

    n_features = len(features)
    nrows = math.ceil(n_features / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))

    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        x_vals = df[feature].values

        ax.scatter(x_vals, logit_pred, alpha=0.4)

        smooth = lowess(logit_pred, x_vals, frac=0.4, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1])

        ax.set_xlabel(feature)
        ax.grid(True)
        ax.set_ylabel("Logit(predicted probability)")
        ax.set_title(f"Logit vs {feature}")

    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def run_non_linearity_diagnostics(
    df: pd.DataFrame,
    features: list[str],
    model,
    X: pd.DataFrame,
) -> None:
    """
    Run non-linearity diagnostics by plotting logit(predicted probability)
    vs selected features.
    """
    plot_logit_vs_feature(df, features, model, X)
