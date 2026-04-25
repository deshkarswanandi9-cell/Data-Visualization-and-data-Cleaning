import pandas as pd
import re

def transform_data(df):
    """
    Phase 3: Data Transformation
    Transforms column names, corrects types, and creates new features.
    
    Args:
        df (pd.DataFrame): The cleaned dataset.
        
    Returns:
        pd.DataFrame: The transformed dataset.
    """
    print("\n--- Phase 3: Data Transformation ---")
    
    if df is None:
        return None
        
    df_trans = df.copy()
    
    # 14. Convert all column names to snake_case
    def to_snake_case(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    df_trans.columns = [to_snake_case(col) for col in df_trans.columns]
    
    # 15. Correct data types (example: ensuring 'survived' is integer if it was object)
    # In Titanic, most are already correct, but we'll ensure numeric types where possible.
    for col in df_trans.columns:
        if df_trans[col].dtype == 'object':
            try:
                # Attempt to convert to numeric if it's purely numbers
                df_trans[col] = pd.to_numeric(df_trans[col])
            except (ValueError, TypeError):
                pass

    # 16. Create derived features
    # Titanic specific: family_size and is_alone
    if 'sibsp' in df_trans.columns and 'parch' in df_trans.columns:
        df_trans['family_size'] = df_trans['sibsp'] + df_trans['parch'] + 1
        df_trans['is_alone'] = (df_trans['family_size'] == 1).astype(int)
        print("Created features: 'family_size', 'is_alone'")

    # 17. Encode binary categorical columns
    for col in df_trans.columns:
        if df_trans[col].dtype == 'object' or df_trans[col].dtype.name == 'category':
            unique_vals = df_trans[col].unique()
            if len(unique_vals) == 2:
                # Simple Label Encoding for binary
                val_map = {val: i for i, val in enumerate(unique_vals)}
                df_trans[col] = df_trans[col].map(val_map)
                print(f"Encoded binary column '{col}': {val_map}")

    # 19. Print final schema
    print("\nFinal Transformed Schema:")
    print(df_trans.dtypes)
    
    return df_trans

if __name__ == "__main__":
    import seaborn as sns
    df = sns.load_dataset('titanic')
    transform_data(df)
