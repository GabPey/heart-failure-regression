import matplotlib.pyplot as plt
import seaborn as sns

# Create a correlation heatmap for the dataframe
def correlation_heatmap(df):
    """
    This function creates a correlation heatmap for the numerical columns in the dataframe.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    
    Returns:
    None: Displays the correlation heatmap.
    """
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

# Look at target variable distribution
def target_distribution(df, target_col):
    """
    This function creates a count plot for the target variable in the dataframe.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    target_col (str): The name of the target column.
    
    Returns:
    None: Displays the count plot for the target variable.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=df[target_col])
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.show()

# Plot side by side the hisograms and boxplots of the variables with high correlation with the target variable
def box_histograms_all(df, target_col, threshold=0.1):
    """
    This function creates side by side histograms and boxplots for the features with high correlation with the target variable.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    target_col (str): The name of the target column.
    threshold (float): The correlation threshold to select features.
    
    Returns:
    None: Displays the histograms and boxplots for the selected features.
    """
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numerical_cols].corr()
    target_corr = corr_matrix[target_col].abs()
    high_corr_features = target_corr[target_corr > threshold].index.drop(target_col)
    
    for feature in high_corr_features:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Histogram of {feature}')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[target_col], y=df[feature])
        plt.title(f'Boxplot of {feature} by {target_col}')
        
        plt.tight_layout()
        plt.show()