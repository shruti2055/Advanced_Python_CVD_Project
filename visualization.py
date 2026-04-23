"""Visualization module.

Create plots to explore dataset features and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(df):
    """Plot the distributions of key numeric variables."""
    print("\n" + "=" * 60)
    print("PLOTTING DISTRIBUTIONS")
    print("=" * 60)

    plot_features = [name for name in ['age', 'height', 'weight', 'ap_hi', 'ap_lo'] if name in df.columns]
    if not plot_features:
        print("No numeric features found for distribution plotting")
        return

    rows = (len(plot_features) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for index, feature_name in enumerate(plot_features):
        axes[index].hist(df[feature_name].dropna(), bins=40, color='skyblue', edgecolor='black', alpha=0.7)
        axes[index].set_title(f'Distribution of {feature_name}')
        axes[index].set_xlabel(feature_name)
        axes[index].set_ylabel('Frequency')
        axes[index].grid(axis='y', alpha=0.3)

    for index in range(len(plot_features), len(axes)):
        axes[index].set_visible(False)

    plt.tight_layout()
    plt.savefig('distribution_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved distribution plots to 'distribution_plots.png'")


def plot_correlation(df):
    """Plot the correlation heatmap for numeric variables."""
    print("\n" + "=" * 60)
    print("PLOTTING CORRELATION HEATMAP")
    print("=" * 60)

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) == 0:
        print("No numeric columns found for correlation plotting")
        return

    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(min(12, len(numeric_columns) * 1.2), min(10, len(numeric_columns) * 1.2)))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved correlation heatmap to 'correlation_heatmap.png'")


def plot_target_distribution(df):
    """Plot the balance of the target class values."""
    print("\n" + "=" * 60)
    print("PLOTTING TARGET DISTRIBUTION")
    print("=" * 60)

    if 'cardio' not in df.columns:
        print("Target column 'cardio' not found")
        return

    counts = df['cardio'].value_counts().sort_index()
    labels = ['No Disease (0)', 'Disease (1)']
    values = [counts.get(0, 0), counts.get(1, 0)]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['steelblue', 'lightcoral'], edgecolor='black', alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        percentage = (height / len(df) * 100) if len(df) > 0 else 0
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom')

    plt.title('Target Variable Distribution')
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved target distribution to 'target_distribution.png'")


def run_all_visualizations(df):
    """Run all exploratory data plots."""
    plot_distributions(df)
    plot_correlation(df)
    plot_target_distribution(df)


def plot_confusion_matrix(cm, model_name):
    """Plot a confusion matrix for a trained model."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrix to 'confusion_matrix.png'")


def plot_f1_comparison(cv_results):
    """Plot the F1-score comparison for all models."""
    if isinstance(cv_results, dict):
        models = list(cv_results.keys())
        f1_values = [cv_results[model]['f1_mean'] for model in models]
        f1_errors = [cv_results[model]['f1_std'] for model in models]
    else:
        models = cv_results['Model'].tolist()
        f1_values = [float(value.split(' ± ')[0]) for value in cv_results['F1-Score'].tolist()]
        f1_errors = [float(value.split(' ± ')[1]) for value in cv_results['F1-Score'].tolist()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, f1_values, yerr=f1_errors, capsize=5, color='skyblue', edgecolor='black')
    plt.title('Model F1-Score Comparison')
    plt.ylabel('F1-Score')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)

    for bar, mean, std in zip(bars, f1_values, f1_errors):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved F1 comparison plot to 'f1_comparison.png'")


def plot_model_comparison(results_df):
    """Plot recall and F1-score for cross-validation results."""
    if results_df.empty:
        print("No results available for model comparison plot")
        return

    models = results_df['Model']
    recall_values = [float(value.split(' ± ')[0]) for value in results_df['Recall'].tolist()]
    f1_values = [float(value.split(' ± ')[0]) for value in results_df['F1-Score'].tolist()]

    plt.figure(figsize=(12, 6))
    positions = range(len(models))
    width = 0.35

    plt.bar([pos - width / 2 for pos in positions], recall_values, width, label='Recall', color='steelblue')
    plt.bar([pos + width / 2 for pos in positions], f1_values, width, label='F1-Score', color='lightgreen')

    plt.xticks(positions, models, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('Cross-Validation Recall versus F1-Score')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved model comparison plot to 'model_comparison.png'")


def plot_threshold_tuning(threshold_df, model_name):
    """Plot how precision, recall, and F1 change with the decision threshold."""
    if threshold_df.empty:
        print("No threshold tuning results to plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['Threshold'], threshold_df['Precision'], marker='o', label='Precision')
    plt.plot(threshold_df['Threshold'], threshold_df['Recall'], marker='o', label='Recall')
    plt.plot(threshold_df['Threshold'], threshold_df['F1-Score'], marker='o', label='F1-Score')
    plt.title(f'Threshold Tuning for {model_name}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(threshold_df['Threshold'])
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    file_name = f'threshold_tuning_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved threshold tuning plot to '{file_name}'")


def plot_feature_importance(importance_df, model_name):
    """Plot the most important features for a tree-based model."""
    if importance_df is None or importance_df.empty:
        print(f"No importance data available for {model_name}")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df.head(10), palette='viridis')
    plt.title(f'Feature Importance - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    file_name = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance to '{file_name}'")

