import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from fpdf import FPDF
import os

def evaluate_and_report(results, trained_models, X_test, y_test, output_path="ML_Report.pdf"):
    """Generates evaluation plots and a final PDF report."""
    if not os.path.exists('output'): os.makedirs('output')
    
    plots = {}
    
    # 1. Confusion Matrix (Random Forest)
    rf_model = trained_models['Random Forest']
    rf_preds = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, rf_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = 'output/confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    plots['Confusion Matrix'] = cm_path

    # 2. ROC Curve
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, rf_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkgreen', label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.legend()
    plt.title('ROC Curve - Random Forest')
    roc_path = 'output/roc_curve.png'
    plt.savefig(roc_path)
    plt.close()
    plots['ROC Curve'] = roc_path

    # 3. PDF Generation
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 20, "Predictive Modeling ML Project Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Model Performance Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    for model, metrics in results.items():
        pdf.cell(0, 8, f"{model}:", ln=True)
        for m, v in metrics.items():
            pdf.cell(0, 8, f"   - {m}: {v:.4f}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Evaluation Visualizations", ln=True)
    
    for name, path in plots.items():
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, name, ln=True)
        pdf.image(path, w=150)
        pdf.ln(10)
        
    pdf.output(output_path)
    print(f"Standalone ML Report saved to: {output_path}")
