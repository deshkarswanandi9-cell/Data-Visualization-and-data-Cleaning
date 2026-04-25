import pandas as pd
import numpy as np

def clean_data(df):
    """
    Phase 2: Data Cleaning
    Handles duplicates, missing values, and outliers.
    
    Args:
        df (pd.DataFrame): The raw dataset.
        
    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    print("\n--- Phase 2: Data Cleaning ---")
    
    if df is None:
        print("No data provided for cleaning.")
        return None
        
    df_clean = df.copy()
    
    # Before stats
    initial_shape = df_clean.shape
    initial_nulls = df_clean.isnull().sum().sum()
    
    # 8. Drop fully duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # 9 & 10. Handle missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if df_clean[col].dtype in ['int64', 'float64']:
                # Fill with median for numerical
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
            else:
                # Fill with mode for categorical
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                else:
                    df_clean[col] = df_clean[col].fillna("Unknown")

    # 11 & 12. Handle outliers using IQR
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    outlier_summary = {}
    
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        
        # Count outliers before capping
        outliers_count = ((df_clean[col] < lower_fence) | (df_clean[col] > upper_fence)).sum()
        if outliers_count > 0:
            outlier_summary[col] = outliers_count
            # Cap outliers
            df_clean[col] = df_clean[col].clip(lower=lower_fence, upper=upper_fence)

    # 13. Before/After comparison
    final_shape = df_clean.shape
    final_nulls = df_clean.isnull().sum().sum()
    
    print(f"Initial Shape: {initial_shape} | Final Shape: {final_shape}")
    print(f"Initial Nulls: {initial_nulls} | Final Nulls: {final_nulls}")
    print(f"Outliers Capped: {outlier_summary}")
    
    return df_clean

if __name__ == "__main__":
    import seaborn as sns
    df = sns.load_dataset('titanic')
    clean_data(df)
