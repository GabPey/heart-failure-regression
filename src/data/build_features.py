"""
This script is responsible for building new features from the raw dataset. 
"""

import numpy as np

def build_features(df):
    """
    This function takes a dataframe as input and returns a dataframe with new features.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    
    Returns:
    pd.DataFrame: The dataframe with new features.
    """
    # Log transformation of serum creatinine
    df["creatinine_log"] = np.log(df["serum_creatinine"])
    # Potential interactions between features

    # Interactions between age and ejection fraction. 
    # Ejection fraction is a measure of how well the heart is pumping blood, and age can affect heart function. 
    # This interaction might capture the combined effect of age and heart function on the likelihood of death.

    # NOTE: after VIF we found strong collinearity between this interaction and the original features, 
    # but we keep it because it has a strong interpretability and is relevant for inference.
    # to reduce collinearity we are going to center the age and ejection fraction variables before creating the interaction.
    df["age_centered"] = df["age"] - df["age"].mean()
    df["ejection_fraction_centered"] = df["ejection_fraction"] - df["ejection_fraction"].mean()
    df["age_ejection_interaction"] = df["age_centered"] * df["ejection_fraction_centered"]

    # Interactions between ejection fraction and serum creatinine. 
    # Serum creatinine is a measure of kidney function, and ejection fraction reflects heart function. 
    # This interaction might capture the combined effect of heart and kidney function on the likelihood of death.
    df["ejection_creatinine_interaction"] = df["ejection_fraction"] * df["serum_creatinine"]

    # Interactions between age and diabetes.
    # Diabetes can affect kidney function and heart health. 
    # This interaction might capture the combined effect of age and diabetes on the likelihood of death.
    df["age_diabetes_interaction"] = df["age"] * df["diabetes"]

    # Interaction between sodium and creatinine.
    # Sodium levels can be affected by kidney function, and creatinine is a measure of kidney function. 
    df["sodium_creatinine_interaction"] = df["serum_sodium"] / df["serum_creatinine"]

    # After creating the interactions, we can drop the original serum_creatinine variable to reduce collinearity, as the log transformation and interactions capture its effect.
    df = df.drop(columns=["serum_creatinine"])
    
    # NOTE: we drop age after centering to reduce collinearity, as the interaction captures its effect.
    df = df.drop(columns=["age"])

    # We also drop ejection fraction after centering to reduce collinearity, as the interaction captures its effect.
    df = df.drop(columns=["ejection_fraction"])
    
    return df

