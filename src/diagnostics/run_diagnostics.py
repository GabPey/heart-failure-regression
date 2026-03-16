import pandas as pd
from sklearn.model_selection import train_test_split

from diagnostics.collinearity import run_collinearity_diagnostics
from diagnostics.non_linearity import run_non_linearity_diagnostics

from models.logistic.prediction import run_logistic_pipeline


def run_raw_diagnostics() -> None:
    """
    Run diagnostics for the raw-feature logistic model.
    """

    print("\n==============================")
    print("RAW MODEL DIAGNOSTICS")
    print("==============================")

    df = pd.read_csv("../data/raw/heart_failure_clinical_records_dataset.csv")

    # Raw features: all predictors except target
    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = run_logistic_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    model = results["model"]

    features_to_plot = [
        "age",
        "ejection_fraction",
        "serum_creatinine",
        "time",
    ]

    run_collinearity_diagnostics(df, features_to_plot)

    run_non_linearity_diagnostics(
        df=df,
        features=features_to_plot,
        model=model,
        X=X,
    )


def run_processed_diagnostics() -> None:
    """
    Run diagnostics for the engineered-feature logistic model.
    """

    print("\n==============================")
    print("ENGINEERED MODEL DIAGNOSTICS")
    print("==============================")

    df = pd.read_csv("../data/processed/heart_failure_clinical_records_dataset_processed.csv")

    features = [
        "time",
        "age_centered",
        "ejection_fraction_centered",
        "sodium_creatinine_interaction",
    ]

    X = df[features]
    y = df["DEATH_EVENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = run_logistic_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    model = results["model"]

    run_collinearity_diagnostics(df, features)

    run_non_linearity_diagnostics(
        df=df,
        features=features,
        model=model,
        X=X,
    )