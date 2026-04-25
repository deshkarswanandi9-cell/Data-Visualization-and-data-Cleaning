"""
Main Entry Point for Data Cleaning & Visualization Dashboard
Runs all phases of the data pipeline.
"""

from src.loader import load_and_inspect
from src.cleaner import clean_data
from src.transformer import transform_data
from src.visualizer import visualize_data
from src.reporter import generate_report
from src.predictor import train_and_evaluate
import pandas as pd
import os

def main():
    print("=" * 50)
    print("  DATA CLEANING & VISUALIZATION DASHBOARD")
    print("=" * 50)

    # 1. Phase 1: Loading & Inspection
    # You can specify a path to a CSV here if you have one, 
    # otherwise it falls back to Seaborn's Titanic dataset.
    raw_df = load_and_inspect()
    
    if raw_df is None:
        print("CRITICAL ERROR: Failed to load data. Exiting.")
        return

    # 2. Phase 2: Cleaning
    cleaned_df = clean_data(raw_df)
    
    # 3. Phase 3: Transformation
    transformed_df = transform_data(cleaned_df)
    
    # Export cleaned dataset
    if not os.path.exists('output'):
        os.makedirs('output')
    transformed_df.to_csv('output/cleaned_dataset.csv', index=False)
    print(f"\n[SUCCESS] Cleaned dataset saved to output/cleaned_dataset.csv")

    # 4. Phase 4: Visualization
    charts = visualize_data(transformed_df)
    
    # 5. Phase 5: Predictive Modeling
    # Default target for Titanic is 'survived' (transformed to 'survived' by snake_case)
    target = 'survived'
    ml_results, ml_plots = train_and_evaluate(transformed_df, target)

    # 6. Phase 6: Reporting
    generate_report(raw_df, transformed_df, charts, ml_results, ml_plots)

    print("\n" + "=" * 50)
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
