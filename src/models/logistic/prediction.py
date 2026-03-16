import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_binary_classifier(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> dict:
    """
    Compute standard classification metrics for binary classification.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def run_logistic_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
    show_confusion_matrix: bool = False,
) -> dict:
    """
    Train and evaluate a Logistic Regression classifier.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    random_state : int
    show_confusion_matrix : bool

    Returns
    -------
    dict
        Standardized result dictionary for model comparison.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=random_state))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions (train)
    y_pred_train = pd.Series(
        pipeline.predict(X_train),
        index=X_train.index,
        name="y_pred",
    )

    y_proba_train = pd.Series(
        pipeline.predict_proba(X_train)[:, 1],
        index=X_train.index,
        name="y_proba",
    )

    # Predictions (test)
    y_pred_test = pd.Series(
        pipeline.predict(X_test),
        index=X_test.index,
        name="y_pred",
    )

    y_proba_test = pd.Series(
        pipeline.predict_proba(X_test)[:, 1],
        index=X_test.index,
        name="y_proba",
    )

    # Metrics
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

    # Optional confusion matrix
    if show_confusion_matrix:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred_test,
        )
        plt.title("Logistic Regression Confusion Matrix")
        plt.tight_layout()
        plt.show()

    return {
        "model_name": "Logistic Regression",
        "model": pipeline,
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