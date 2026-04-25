import os
import datetime
from fpdf import FPDF

def export_as_pdf(df_raw, df_clean, charts_list, output_path):
    """Generates a comprehensive PDF report with data metrics, cleaning logs, and charts."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Title Page ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 40, "Data Cleaning & Visualization Report", ln=True, align="C")
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 10, "Dataset: Titanic (Seaborn)", ln=True, align="C")
    pdf.ln(20)
    
    # --- 1. Data Inspection (Phase 1) ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1. Data Inspection & Initial State", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    pdf.cell(0, 10, f"Original Dataset Shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns", ln=True)
    pdf.cell(0, 10, f"Duplicate Rows Found: {df_raw.duplicated().sum()}", ln=True)
    pdf.ln(5)
    
    # Missing Values Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Missing Values Summary:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    null_counts = df_raw.isnull().sum()
    null_data = [["Column", "Null Count", "Percentage"]]
    for col, count in null_counts.items():
        if count > 0:
            null_data.append([col, str(count), f"{(count/len(df_raw)*100):.2f}%"])
    
    with pdf.table() as table:
        for data_row in null_data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)
    pdf.ln(10)
    
    # --- 2. Descriptive Statistics (Phase 1) ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2. Descriptive Statistics", ln=True)
    pdf.ln(5)
    
    stats = df_raw.describe(include='all').transpose()
    stats = stats.reset_index().rename(columns={'index': 'Feature'})
    # Take key columns to fit PDF width
    key_stats = ['Feature', 'count', 'unique', 'top', 'mean', 'std', 'min', 'max']
    stats_subset = stats[key_stats].fillna("-")
    
    stats_data = [key_stats]
    for _, row_data in stats_subset.iterrows():
        stats_data.append([str(val)[:15] for val in row_data.values]) # Truncate long strings
        
    with pdf.table(width=190) as table:
        for data_row in stats_data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)
    pdf.ln(10)
    
    # --- 3. Data Cleaning & Transformation (Phase 2 & 3) ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "3. Processing & Transformation Logs", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    pdf.cell(0, 10, f"Cleaned Dataset Shape: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns", ln=True)
    pdf.cell(0, 10, "Outlier Handling: IQR method used to cap numerical outliers.", ln=True)
    
    new_features = [col for col in df_clean.columns if col not in [c.lower() for c in df_raw.columns]]
    pdf.cell(0, 10, f"New Features Created: {', '.join(new_features)}", ln=True)
    
    # Encoding info
    encoded_cols = []
    for col in df_clean.columns:
        if df_clean[col].dtype == 'int64' and col in df_raw.columns and df_raw[col].dtype == 'object':
            encoded_cols.append(col)
    if encoded_cols:
        pdf.cell(0, 10, f"Encoded Categorical Columns: {', '.join(encoded_cols)}", ln=True)
    
    pdf.ln(10)
    
    # --- 4. Visualizations (Phase 4) ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "4. Data Visualizations", ln=True)
    pdf.ln(5)
    
    for chart_path in charts_list:
        chart_name = os.path.basename(chart_path).replace(".png", "").replace("_", " ").title()
        
        # Check if we need a new page for the next chart
        if pdf.get_y() > 180:
            pdf.add_page()
            
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, chart_name, ln=True)
        
        # Add Image
        pdf.image(chart_path, x=15, w=180)
        pdf.ln(10)
        
    try:
        pdf.output(output_path)
        print(f"PDF Report saved to: {output_path}")
    except Exception as e:
        print(f"Failed to export PDF report: {e}")

def generate_report(df_raw, df_clean, charts_list, output_path_html="output/report.html", output_path_pdf="output/report.pdf"):
    """
    Phase 5: Reporting & Summary
    Prints a console report and exports both HTML and PDF reports.
    """
    print("\n--- Phase 5: Reporting & Summary ---")
    
    if df_raw is None or df_clean is None:
        print("Missing data to generate report.")
        return

    # Metadata
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = "Titanic Dataset"
    
    # Original vs Cleaned shape
    raw_shape = df_raw.shape
    clean_shape = df_clean.shape
    
    # Duplicates
    duplicates_removed = df_raw.duplicated().sum()
    
    # Missing values handled
    null_cols = df_raw.columns[df_raw.isnull().any()].tolist()
    
    # New features
    new_features = [col for col in df_clean.columns if col not in [c.lower() for c in df_raw.columns]]

    # --- Console Report ---
    print(f"Dataset Name: {dataset_name}")
    print(f"Report Generated: {report_date}")
    print("-" * 30)
    print(f"Original Shape: {raw_shape}")
    print(f"Cleaned Shape: {clean_shape}")
    print(f"Duplicates Removed: {duplicates_removed}")
    print(f"Columns with Nulls Handled: {', '.join(null_cols)}")
    print(f"Outliers: Detected and capped using IQR method.")
    print(f"New Features Created: {', '.join(new_features)}")
    print(f"Total Charts Generated: {len(charts_list)}")
    print("-" * 30)

    # --- Export HTML ---
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Cleaning & Visualization Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 20px; background-color: #f4f7f6; }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .summary-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }}
            .stat-item {{ background: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }}
            .stat-value {{ font-size: 1.5em; font-weight: bold; color: #3498db; }}
            .chart-container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
            .chart-item {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 100%; max-width: 450px; }}
            .chart-item img {{ width: 100%; height: auto; border-radius: 5px; }}
            footer {{ text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Data Cleaning & Visualization Report</h1>
        <p><strong>Dataset:</strong> {dataset_name} | <strong>Date:</strong> {report_date}</p>
        
        <div class="summary-card">
            <h2>Pipeline Summary</h2>
            <div class="stats-grid">
                <div class="stat-item"><div class="stat-value">{raw_shape[0]}</div>Original Rows</div>
                <div class="stat-item"><div class="stat-value">{clean_shape[0]}</div>Cleaned Rows</div>
                <div class="stat-item"><div class="stat-value">{duplicates_removed}</div>Duplicates Removed</div>
                <div class="stat-item"><div class="stat-value">{len(new_features)}</div>New Features</div>
            </div>
            <p style="margin-top:20px;">
                <strong>Cleaning Methods:</strong> Missing numerical values were filled with medians. 
                Categorical values were filled with modes. Outliers were capped using the Interquartile Range (IQR) method.
            </p>
        </div>

        <h2>Visualizations</h2>
        <div class="chart-container">
            {"".join([f'<div class="chart-item"><h3>{os.path.basename(path).replace(".png", "").replace("_", " ").title()}</h3><img src="charts/{os.path.basename(path)}" alt="Chart"></div>' for path in charts_list])}
        </div>

        <footer>
            Generated by Antigravity AI Data Pipeline &copy; 2026
        </footer>
    </body>
    </html>
    """
    
    try:
        with open(output_path_html, "w") as f:
            f.write(html_template)
        print(f"HTML Report saved to: {output_path_html}")
    except Exception as e:
        print(f"Failed to export HTML report: {e}")

    # --- Export PDF ---
    export_as_pdf(df_raw, df_clean, charts_list, output_path_pdf)


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({'a': [1,2], 'b': [3,4]})
    generate_report(df, df, [])
