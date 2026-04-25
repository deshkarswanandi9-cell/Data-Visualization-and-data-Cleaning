import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

def get_data(target='survived'):
    """Loads dataset and performs minimal cleaning for ML."""
    print(f"Loading dataset for Predictive Modeling...")
    df = sns.load_dataset('titanic')
    
    # Minimal cleaning needed for ML
    df = df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'])
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    
    # Feature Engineering
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    # Encoding
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['embarked'], drop_first=True)
    
    X = df.drop(columns=[target])
    y = df[target]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
