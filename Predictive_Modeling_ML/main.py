from src.data_handler import get_data
from src.trainer import train_models
from src.evaluator import evaluate_and_report

def main():
    print("="*50)
    print("   PREDICTIVE MODELING USING MACHINE LEARNING")
    print("="*50)
    
    # 1. Load and Split Data
    X_train, X_test, y_train, y_test = get_data(target='survived')
    
    # 2. Train Models
    results, trained_models = train_models(X_train, y_train, X_test, y_test)
    
    # 3. Evaluate and Generate Report
    evaluate_and_report(results, trained_models, X_test, y_test)
    
    print("\n[SUCCESS] Standalone ML Pipeline Completed!")
    print("="*50)

if __name__ == "__main__":
    main()
