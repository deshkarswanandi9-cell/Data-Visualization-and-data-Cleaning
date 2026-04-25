# Data Cleaning & Visualization Dashboard

A modular Python-based data pipeline designed to clean, transform, and visualize datasets. By default, it uses the **Titanic** dataset to demonstrate real-world data quality handling.

## Features
- **Phase 1: Loading & Inspection** - Automated dataset inspection and statistics.
- **Phase 2: Cleaning** - Duplicates removal, median/mode imputation, and IQR-based outlier capping.
- **Phase 3: Transformation** - Column snake_casing, type correction, and feature engineering (`family_size`, `is_alone`).
- **Phase 4: Visualization** - 7+ publication-ready charts (Bar, Histogram, Line, Pie, Box, Scatter, Heatmap).
- **Phase 5: Reporting** - Detailed console summary and a professional HTML report.

## Project Structure
```text
data_viz_project/
├── data/            # Raw datasets (optional)
├── output/          # Pipeline outputs
│   ├── charts/      # PNG visualizations
│   ├── cleaned_dataset.csv
│   └── report.html  # Interactive summary
├── src/             # Source modules
│   ├── loader.py
│   ├── cleaner.py
│   ├── transformer.py
│   ├── visualizer.py
│   └── reporter.py
└── main.py          # Entry point
```

## Setup & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**:
   ```bash
   python main.py
   ```

3. **View Results**:
   - Check the terminal for the console report.
   - Open `output/report.html` in any browser to see the visualizations.

## Tech Stack
- **Python 3.8+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn**: Data visualization
- **Jinja2**: Templating (optional)

---
*Created with Antigravity AI*
