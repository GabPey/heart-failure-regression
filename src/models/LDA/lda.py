from __future__ import annotations

from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
from sklearn.model_selection import train_test_split


def _validate_inputs(df: pd.DataFrame, features: list[str], target: str) -> None:
    """
    Validate that the dataframe contains the requested features and target.
    """
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    if df[target].nunique() != 2:
        raise ValueError("This module is designed for binary classification only.")


def evaluate_binary_classifier(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> dict[str, Any]:
    """
    Compute standard binary classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def plot_class_kdes(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    max_features: int = 4,
) -> None:
    """
    Plot class-wise density-like histograms for selected features.
    """
    features_to_plot = features[:max_features]
    n_features = len(features_to_plot)

    if n_features == 0:
        raise ValueError("No features provided for KDE/histogram plotting.")

    ncols = 2
    nrows = (n_features + ncols - 1) // ncols
    classes = sorted(df[target].unique())

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]

        for cls in classes:
            x = df.loc[df[target] == cls, feature].dropna()
            ax.hist(
                x,
                bins=25,
                density=True,
                alpha=0.45,
                label=f"{target}={cls}",
            )

        ax.set_title(f"Class distributions: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()

    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def plot_pairwise_scatter(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    max_pairs: int = 2,
) -> None:
    """
    Plot a few pairwise scatter plots colored by class.
    """
    if len(features) < 2:
        print("Skipping pairwise scatter: at least 2 features are required.")
        return

    classes = sorted(df[target].unique())
    pairs = list(combinations(features, 2))[:max_pairs]

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5))
    axes = [axes] if n_pairs == 1 else axes

    for ax, (x_feat, y_feat) in zip(axes, pairs):
        for cls in classes:
            subset = df[df[target] == cls]
            ax.scatter(
                subset[x_feat],
                subset[y_feat],
                alpha=0.6,
                label=f"{target}={cls}",
            )

        ax.set_title(f"{y_feat} vs {x_feat}")
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.legend()

    plt.tight_layout()
    plt.show()


def fit_lda(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    solver: str = "svd",
) -> LinearDiscriminantAnalysis:
    """
    Fit an LDA classifier.
    """
    model = LinearDiscriminantAnalysis(solver=solver)
    model.fit(X_train, y_train)
    return model


def plot_lda_projection(
    model: LinearDiscriminantAnalysis,
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
) -> None:
    """
    Plot the distribution of the first LDA discriminant component by class.
    """
    lda_scores = model.transform(X)

    if lda_scores.shape[1] == 0:
        print("Skipping LDA projection plot: no discriminant component available.")
        return

    classes = sorted(y.unique())

    plt.figure(figsize=(8, 5))
    for cls in classes:
        x = lda_scores[y == cls, 0]
        plt.hist(
            x,
            bins=25,
            density=True,
            alpha=0.45,
            label=f"{target}={cls}",
        )

    plt.title("Distribution of first LDA component")
    plt.xlabel("LD1")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series,
) -> None:
    """
    Plot confusion matrix for test predictions.
    """
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("LDA Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    y_test: pd.Series,
    y_proba: pd.Series,
) -> None:
    """
    Plot ROC curve for test predictions.
    """
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("LDA ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.tight_layout()
    plt.show()


def run_lda_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    solver: str = "svd",
    show_confusion_matrix: bool = False,
    show_roc_curve: bool = False,
    show_projection: bool = False,
    target_name: str = "DEATH_EVENT",
) -> dict[str, Any]:
    """
    Train and evaluate LDA using a pre-defined train/test split.

    This is the standardized function to use when comparing LDA against
    Logistic Regression, KNN, and Naive Bayes.
    """
    model = fit_lda(X_train=X_train, y_train=y_train, solver=solver)

    y_pred_train = pd.Series(
        model.predict(X_train),
        index=X_train.index,
        name="y_pred",
    )
    y_proba_train = pd.Series(
        model.predict_proba(X_train)[:, 1],
        index=X_train.index,
        name="y_proba",
    )

    y_pred_test = pd.Series(
        model.predict(X_test),
        index=X_test.index,
        name="y_pred",
    )
    y_proba_test = pd.Series(
        model.predict_proba(X_test)[:, 1],
        index=X_test.index,
        name="y_proba",
    )

    train_metrics = evaluate_binary_classifier(
        y_train,
        y_pred_train,
        y_proba_train,
    )
    test_metrics = evaluate_binary_classifier(
        y_test,
        y_pred_test,
        y_proba_test,
    )

    if show_projection:
        plot_lda_projection(model=model, X=X_train, y=y_train, target=target_name)

    if show_confusion_matrix:
        plot_confusion_matrix(y_test=y_test, y_pred=y_pred_test)

    if show_roc_curve:
        plot_roc_curve(y_test=y_test, y_proba=y_proba_test)

    return {
        "model_name": "LDA",
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
    }


def run_lda_diagnostics(
    df: pd.DataFrame,
    features: list[str],
    target: str = "DEATH_EVENT",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    solver: str = "svd",
    max_kde_features: int = 4,
    max_scatter_pairs: int = 2,
) -> dict[str, object]:
    """
    Run the full LDA diagnostic workflow.

    Steps
    -----
    1. Plot class-wise distributions for selected features
    2. Plot pairwise scatter plots for selected features
    3. Split data into train and test sets
    4. Fit LDA
    5. Plot distribution of the first LDA component
    6. Plot confusion matrix
    7. Plot ROC curve
    """
    _validate_inputs(df=df, features=features, target=target)

    X = df[features].copy()
    y = df[target].copy()

    stratify_y = y if stratify else None

    plot_class_kdes(
        df=df,
        features=features,
        target=target,
        max_features=max_kde_features,
    )
    plot_pairwise_scatter(
        df=df,
        features=features,
        target=target,
        max_pairs=max_scatter_pairs,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    results = run_lda_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        solver=solver,
        show_confusion_matrix=True,
        show_roc_curve=True,
        show_projection=True,
        target_name=target,
    )

    return results