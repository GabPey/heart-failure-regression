# We try different feature selection methods to improve our model

# We start with backwards selection based on the p-values of the features in the logistic regression model. 
# We will iteratively remove the least significant feature until all remaining features are statistically significant.

import pandas as pd 
import statsmodels.api as sm

def backward_selection():
    df = pd.read_csv("../data/raw/heart_failure_clinical_records_dataset.csv")
    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(maxiter=1000)
    p_values = model.pvalues
    while p_values.max() > 0.05:
        feature_to_remove = p_values.idxmax()
        X = X.drop(columns=[feature_to_remove])
        model = sm.Logit(y, X).fit(maxiter=1000)
        p_values = model.pvalues
    print(model.summary())
    return model

def backward_selection_fe():
    df = pd.read_csv("../data/processed/heart_failure_clinical_records_dataset_processed.csv")
    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(maxiter=1000)
    p_values = model.pvalues
    while p_values.max() > 0.05:
        feature_to_remove = p_values.idxmax()
        X = X.drop(columns=[feature_to_remove])
        model = sm.Logit(y, X).fit(maxiter=1000)
        p_values = model.pvalues
    print(model.summary())
    return model