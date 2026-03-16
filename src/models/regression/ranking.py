import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def rank_targets_by_r2(df):

    numeric_cols = df.select_dtypes(include=["number"]).columns
    results = []

    for target in numeric_cols:

        X = df.drop(columns=[target])
        X = X.select_dtypes(include=["number"])

        y = df[target]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge())
        ])

        scores = cross_val_score(model, X, y, cv=5, scoring="r2")

        results.append({
            "target": target,
            "mean_r2": np.mean(scores),
            "std_r2": np.std(scores)
        })

    return pd.DataFrame(results).sort_values("mean_r2", ascending=False)
