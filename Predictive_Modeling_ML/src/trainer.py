from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def train_models(X_train, y_train, X_test, y_test):
    """Trains Linear Regression, Decision Tree, and Random Forest."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        if name == "Linear Regression":
            results[name] = {
                "MSE": mean_squared_error(y_test, preds),
                "R2": r2_score(y_test, preds)
            }
        else:
            results[name] = {
                "Accuracy": accuracy_score(y_test, preds)
            }
        
        trained_models[name] = model
        
    return results, trained_models
