import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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


def plot_boxplot_hist_by_skew(
    df,
    exclude=None,
    skew_threshold=1.0,
    bins=30,
    figsize_per_row=(12, 4),
    iqr_factor=1.5
):
    """
    Plot (boxplot, histogram + KDE) side by side for continuous variables
    with absolute skewness above a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    exclude : list or None
        Columns to exclude (e.g. target).
    skew_threshold : float
        Plot only variables with abs(skewness) >= this threshold.
    bins : int
        Number of bins in histogram.
    figsize_per_row : tuple
        Figure size per row: (width, height).
    iqr_factor : float
        Multiplier for IQR rule to define outliers.
    """

    exclude = exclude or []

    # ----------------------------
    # 1) Identify continuous vars
    # ----------------------------
    continuous_vars = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 2:
            continuous_vars.append(col)

    # ----------------------------
    # 2) Select skewed vars
    # ----------------------------
    selected_vars = []
    skew_values = {}

    for col in continuous_vars:
        s = df[col].dropna()
        skew_val = s.skew()
        if abs(skew_val) >= skew_threshold:
            selected_vars.append(col)
            skew_values[col] = skew_val

    if not selected_vars:
        print(f"No continuous variables found with |skew| >= {skew_threshold}.")
        return

    # ----------------------------
    # 3) Create figure
    # ----------------------------
    n = len(selected_vars)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n)
    )

    if n == 1:
        axes = np.array([axes])

    # ----------------------------
    # 4) Plot each variable
    # ----------------------------
    for i, col in enumerate(selected_vars):
        ax_box = axes[i, 0]
        ax_hist = axes[i, 1]

        data = df[col].dropna().astype(float)

        # IQR outliers
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr

        outliers = data[(data < lower) | (data > upper)]
        non_outliers = data[(data >= lower) & (data <= upper)]

        # -------- Boxplot --------
        bp = ax_box.boxplot(
            data,
            vert=True,
            patch_artist=True,
            showfliers=False,
            widths=0.5
        )

        # Style boxplot elements
        for box in bp["boxes"]:
            box.set(facecolor="#d9eaf7", edgecolor="#4c4c4c", linewidth=1.2)
        for whisker in bp["whiskers"]:
            whisker.set(color="#4c4c4c", linewidth=1.2)
        for cap in bp["caps"]:
            cap.set(color="#4c4c4c", linewidth=1.2)
        for median in bp["medians"]:
            median.set(color="#d62728", linewidth=1.6)

        # Plot outliers manually with color
        if len(outliers) > 0:
            x_jitter = np.random.normal(loc=1, scale=0.03, size=len(outliers))
            ax_box.scatter(
                x_jitter,
                outliers,
                alpha=0.8,
                s=28,
                color="#ff7f0e",
                edgecolor="black",
                linewidth=0.4,
                label="Outliers"
            )
            ax_box.legend(frameon=False, loc="upper right")

        ax_box.set_title(
            f"{col}\nBoxplot | skew={skew_values[col]:.2f} | outliers={len(outliers)}",
            fontsize=11
        )
        ax_box.set_ylabel(col)
        ax_box.set_xticks([])
        ax_box.grid(axis="y", alpha=0.25)

        # -------- Histogram --------
        ax_hist.hist(
            data,
            bins=bins,
            density=True,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.6
        )

        # KDE with pandas/matplotlib
        data.plot.kde(ax=ax_hist, linewidth=2)

        # Optional: show outlier thresholds
        ax_hist.axvline(lower, linestyle="--", linewidth=1)
        ax_hist.axvline(upper, linestyle="--", linewidth=1)

        ax_hist.set_title(
            f"{col}\nHistogram + KDE",
            fontsize=11
        )
        ax_hist.set_xlabel(col)
        ax_hist.set_ylabel("Density")
        ax_hist.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()

def outlier_skew_table(
    df,
    exclude=None,
    skew_threshold=1.0,
    iqr_factor=1.5
):
    """
    Generate a summary table with skewness and IQR-based outliers
    for continuous variables.

    Returns
    -------
    pd.DataFrame
    """

    exclude = exclude or []

    results = []

    for col in df.columns:

        if col in exclude:
            continue

        # Only numeric & not binary
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) <= 2:
            continue

        data = df[col].dropna().astype(float)

        # --- Skew ---
        skew_val = data.skew()

        # --- IQR ---
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr

        outliers = data[(data < lower) | (data > upper)]
        n_outliers = len(outliers)
        pct_outliers = n_outliers / len(data)

        results.append({
            "variable": col,
            "skewness": round(skew_val, 3),
            "abs_skew": round(abs(skew_val), 3),
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "lower_bound": round(lower, 3),
            "upper_bound": round(upper, 3),
            "n_outliers": n_outliers,
            "pct_outliers": round(pct_outliers, 3),
            "skew_flag": abs(skew_val) >= skew_threshold,
            "outlier_flag": pct_outliers > 0.05,  # 5% threshold (ajustable)
            "problematic": (abs(skew_val) >= skew_threshold) or (pct_outliers > 0.05)
        })

    summary = pd.DataFrame(results)

    # Ordena por gravedad
    summary = summary.sort_values(
        by=["problematic", "abs_skew", "pct_outliers"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return summary