import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def visualize_data(df, output_dir="output/charts"):
    """
    Phase 4: Visualization
    Generates and saves 7+ charts.
    
    Args:
        df (pd.DataFrame): The cleaned and transformed dataset.
        output_dir (str): Directory to save charts.
    """
    print("\n--- Phase 4: Visualization ---")
    
    if df is None:
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sns.set_theme(style="whitegrid")
    palette = "Blues_d"
    saved_files = []

    # 1. Bar Chart: Count of passengers per class
    plt.figure(figsize=(10, 6))
    if 'pclass' in df.columns:
        sns.countplot(data=df, x='pclass', palette='Set2')
        plt.title('Passenger Count per Class')
        plt.xlabel('Class')
        plt.ylabel('Count')
        path = os.path.join(output_dir, 'bar_passenger_class.png')
        plt.savefig(path, dpi=150)
        saved_files.append(path)
        plt.close()

    # 2. Histogram: Distribution of Age with KDE
    plt.figure(figsize=(10, 6))
    if 'age' in df.columns:
        sns.histplot(data=df, x='age', kde=True, color='skyblue')
        plt.title('Age Distribution of Passengers')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        path = os.path.join(output_dir, 'histogram_age.png')
        plt.savefig(path, dpi=150)
        saved_files.append(path)
        plt.close()

    # 3. Line Plot: Survival Rate by Age (Trend)
    plt.figure(figsize=(10, 6))
    if 'age' in df.columns and 'survived' in df.columns:
        # Group by rounded age to see trend
        df_trend = df.copy()
        df_trend['age_group'] = (df_trend['age'] // 5) * 5
        survival_trend = df_trend.groupby('age_group')['survived'].mean()
        sns.lineplot(x=survival_trend.index, y=survival_trend.values, marker='o', color='teal')
        plt.title('Survival Rate Trend by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Survival Rate')
        path = os.path.join(output_dir, 'line_survival_trend.png')
        plt.savefig(path, dpi=150)
        saved_files.append(path)
        plt.close()

    # 4. Pie Chart: Proportion of Survival
    plt.figure(figsize=(10, 6))
    if 'survived' in df.columns:
        survival_counts = df['survived'].value_counts()
        plt.pie(survival_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=140)
        plt.title('Survival Proportion')
        path = os.path.join(output_dir, 'pie_survival.png')
        plt.savefig(path, dpi=150)
        saved_files.append(path)
        plt.close()

    # 5. Box Plot: Age by Survival Status
    plt.figure(figsize=(10, 6))
    if 'age' in df.columns and 'survived' in df.columns:
        sns.boxplot(data=df, x='survived', y='age', palette='Set2')
        plt.title('Age Spread by Survival Status')
        plt.xlabel('Survived (0=No, 1=Yes)')
        plt.ylabel('Age')
        path = os.path.join(output_dir, 'boxplot_age_survival.png')
        plt.savefig(path, dpi=150)
        saved_files.append(path)
        plt.close()

    # 6. Scatter Plot: Age vs Fare, colored by Survival
    plt.figure(figsize=(10, 6))
    if 'age' in df.columns and 'fare' in df.columns and 'survived' in df.columns:
        sns.scatterplot(data=df, x='age', y='fare', hue='survived', palette='viridis', alpha=0.6)
        plt.title('Relationship: Age vs Fare (Colored by Survival)')
        plt.xlabel('Age')
        plt.ylabel('Fare')
        path = os.path.join(output_dir, 'scatter_age_fare.png')
        plt.savefig(path, dpi=150)
        saved_files.append(path)
        plt.close()

    # 7. Heatmap: Correlation Matrix
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    path = os.path.join(output_dir, 'heatmap_correlation.png')
    plt.savefig(path, dpi=150)
    saved_files.append(path)
    plt.close()

    print(f"Generated {len(saved_files)} charts in {output_dir}/")
    return saved_files

if __name__ == "__main__":
    import seaborn as sns
    df = sns.load_dataset('titanic')
    # Mock some cleaning/trans
    visualize_data(df)
