import pandas as pd
import seaborn as sns
import os

def load_and_inspect(filepath=None):
    """
    Phase 1: Data Loading & Inspection
    Loads the dataset and prints basic inspection metrics.
    
    Args:
        filepath (str, optional): Path to the CSV file. If None, loads Titanic from seaborn.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    print("--- Phase 1: Data Loading & Inspection ---")
    
    try:
        if filepath and os.path.exists(filepath):
            print(f"Loading dataset from: {filepath}")
            df = pd.read_csv(filepath)
        else:
            print("No CSV found or provided. Falling back to seaborn 'titanic' dataset.")
            df = sns.load_dataset('titanic')
            
        # 2. Print shape
        print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        
        # 3. First 5 rows
        print("\nFirst 5 Rows:")
        print(df.head())
        
        # 4. Column names and types
        print("\nColumn Information:")
        print(df.info())
        
        # 5. Missing values
        print("\nMissing Values Summary:")
        null_counts = df.isnull().sum()
        null_pct = (df.isnull().sum() / len(df)) * 100
        null_df = pd.DataFrame({'Counts': null_counts, 'Percentage (%)': null_pct})
        print(null_df[null_df['Counts'] > 0])
        
        # 6. Duplicates
        duplicates = df.duplicated().sum()
        print(f"\nNumber of Duplicate Rows: {duplicates}")
        
        # 7. Descriptive statistics
        print("\nDescriptive Statistics:")
        print(df.describe(include='all'))
        
        return df
        
    except Exception as e:
        print(f"Error during data loading: {e}")
        return None

if __name__ == "__main__":
    load_and_inspect()
