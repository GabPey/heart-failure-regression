# We create a baseline model for further comparison. 
# We will use a logistic regression model as our baseline, which is a common choice for binary classification problems like ours (predicting death events).

import statsmodels.api as sm
import pandas as pd

# We start by a baseline 
def baseline_logistic():
    # Load the raw data
    df = pd.read_csv("../data/raw/heart_failure_clinical_records_dataset.csv")
    # Define the target variable and features
    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]
    # Add a constant to the features (for the intercept term in logistic regression)
    X = sm.add_constant(X)
    # Fit the logistic regression model
    model = sm.Logit(y, X).fit(maxiter=1000)
    # Print the summary of the model
    print(model.summary())

# Now with the feature engineering
def baseline_logistic_fe():
    # Load the processed data with feature engineering
    df = pd.read_csv("../data/processed/heart_failure_clinical_records_dataset_processed.csv")
    # Define the target variable and features
    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]
    # Add a constant to the features (for the intercept term in logistic regression)
    X = sm.add_constant(X)
    # Fit the logistic regression model
    model = sm.Logit(y, X).fit(maxiter=1000)
    # Print the summary of the model
    print(model.summary())

# A third model using the result of backward selection but adding the age variable to respect hierarchy.
def baseline_logistic_hierarchy():
    df = pd.read_csv("../data/processed/heart_failure_clinical_records_dataset_processed.csv")
    X = df[["ejection_fraction", "creatinine_log", "time", "age", "age_ejection_interaction", "ejection_creatinine_interaction"]]
    y = df["DEATH_EVENT"]
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(maxiter=1000)
    print(model.summary())