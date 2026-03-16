import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix for the given feature matrix."""
    return X.corr()


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted VIF table for the given feature matrix."""
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)


def plot_correlation_matrix(X: pd.DataFrame, title: str = "Correlation Matrix") -> None:
    """Plot the correlation matrix for the given feature matrix."""
    corr = correlation_matrix(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    plt.title(title, pad=20)
    plt.tight_layout()
    plt.show()


def run_collinearity_diagnostics(
    df: pd.DataFrame,
    features: list[str],
    plot: bool = True,
    title: str = "Correlation Matrix",
) -> None:
    """
    Run collinearity diagnostics on the selected features from a dataframe.
    """
    X = df[features].copy()

    print("\n=== VIF TABLE ===")
    print(vif_table(X))

    print("\n=== CORRELATION MATRIX ===")
    print(correlation_matrix(X))

    if plot:
        plot_correlation_matrix(X, title=title)
