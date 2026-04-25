import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
import os

def preprocess_for_ml(df, target_col):
    """
    Prepares the dataset for machine learning by encoding categorical variables
    and dropping non-essential columns.
    """
    print(f"\n--- Preprocessing for ML (Target: {target_col}) ---")
    
    ml_df = df.copy()
    
    # 1. Drop columns that are unlikely to be useful for generic prediction
    # For Titanic: 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'class' 
    # are often redundant if we have 'sex', 'embarked', 'pclass', 'survived'
    redundant_cols = ['who', 'adult_male', 'deck', 'embark_town', 'alive', 'class', 'embark_town']
    cols_to_drop = [c for c in redundant_cols if c in ml_df.columns and c != target_col]
    if cols_to_drop:
        ml_df = ml_df.drop(columns=cols_to_drop)
        print(f"Dropped redundant columns: {cols_to_drop}")

    # 2. Encode remaining categorical columns (One-Hot Encoding)
    categorical_cols = ml_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    if categorical_cols:
        print(f"One-Hot Encoding columns: {categorical_cols}")
        ml_df = pd.get_dummies(ml_df, columns=categorical_cols, drop_first=True)

    # 3. Handle target encoding if it's categorical
    if ml_df[target_col].dtype == 'object' or ml_df[target_col].dtype.name == 'category':
        ml_df[target_col] = pd.factorize(ml_df[target_col])[0]
        print(f"Encoded target column '{target_col}'")

    return ml_df

def train_and_evaluate(df, target_col):
    """
    Trains multiple models and evaluates their performance.
    """
    print("\n--- Phase 4: Predictive Modeling ---")
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return None

    # Preprocess
    ml_df = preprocess_for_ml(df, target_col)
    
    X = ml_df.drop(columns=[target_col])
    y = ml_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    plots = {}
    
    if not os.path.exists('output/ml'):
        os.makedirs('output/ml')

    # --- 1. Linear Regression (Requested) ---
    # Note: Even if it's classification, we'll run it as requested, but also run Logistic
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_preds = lin_reg.predict(X_test)
    results['Linear Regression'] = {
        'MSE': mean_squared_error(y_test, lin_preds),
        'R2': r2_score(y_test, lin_preds)
    }

    # --- 2. Decision Tree ---
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_preds = dt_clf.predict(X_test)
    results['Decision Tree'] = {
        'Accuracy': accuracy_score(y_test, dt_preds)
    }

    # --- 3. Random Forest ---
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_test)
    rf_probs = rf_clf.predict_proba(X_test)[:, 1]
    results['Random Forest'] = {
        'Accuracy': accuracy_score(y_test, rf_preds)
    }

    # --- Evaluation Visuals (Using Random Forest as representative) ---
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, rf_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Random Forest)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = 'output/ml/confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    plots['confusion_matrix'] = cm_path

    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, rf_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = 'output/ml/roc_curve.png'
    plt.savefig(roc_path)
    plt.close()
    plots['roc_curve'] = roc_path
    
    results['Random Forest']['AUC'] = roc_auc

    print("\nModel Results Summary:")
    for model, metrics in results.items():
        print(f"{model}: {metrics}")

    return results, plots

if __name__ == "__main__":
    import seaborn as sns
    from loader import load_and_inspect
    from cleaner import clean_data
    from transformer import transform_data
    
    df = sns.load_dataset('titanic')
    df = clean_data(df)
    df = transform_data(df)
    train_and_evaluate(df, 'survived')
