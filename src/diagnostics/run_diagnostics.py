import pandas as pd

from diagnostics.collinearity import run_collinearity_diagnostics
from diagnostics.non_linearity import run_non_linearity_diagnostics

from models.prediction import (
    predict_with_raw_features,
    predict_with_inference_features
)

def run_raw_diagnostics():

    print("\n==============================")
    print("RAW MODEL DIAGNOSTICS")
    print("==============================")

    df = pd.read_csv("../data/raw/heart_failure_clinical_records_dataset.csv")

    results = predict_with_raw_features()
    model = results["model"]

    # mismas features usadas por el modelo
    X = df.drop(columns=["DEATH_EVENT"])

    features_to_plot = [
        "age",
        "ejection_fraction",
        "serum_creatinine",
        "time"
    ]

    run_collinearity_diagnostics(df, features_to_plot)

    run_non_linearity_diagnostics(
        df=df,
        features=features_to_plot,
        model=model,
        X=X
    )


def run_processed_diagnostics():

    print("\n==============================")
    print("ENGINEERED MODEL DIAGNOSTICS")
    print("==============================")

    df = pd.read_csv("../data/processed/heart_failure_clinical_records_dataset_processed.csv")

    results = predict_with_inference_features()
    model = results["model"]

    features = ["time", "age_centered", "ejection_fraction_centered", "sodium_creatinine_interaction"]
    X = df[features]

    run_collinearity_diagnostics(df, features)

    run_non_linearity_diagnostics(
        df=df,
        features=features,
        model=model,
        X=X
    )