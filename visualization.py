"""Visualization Module.

Creates plots to explore the data.
Shows distributions, correlations, and the target variable.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(df):
    """
    Show histograms for key numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print("\n" + "=" * 60)
    print("PLOTTING DISTRIBUTIONS")
    print("=" * 60)

    feature_list = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    available_features = [f for f in feature_list if f in df.columns]

    if len(available_features) == 0:
        print("No expected numeric features available for distribution plots")
        return

    n = len(available_features)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    fig.suptitle('Distribution of Key Features', fontsize=16, fontweight='bold')
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for idx, feature_name in enumerate(available_features):
        axes[idx].hist(df[feature_name].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {feature_name}', fontweight='bold')
        axes[idx].set_xlabel(feature_name)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(axis='y', alpha=0.3)

    for idx in range(len(available_features), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('distribution_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Distribution plots saved as 'distribution_plots.png'")


def plot_correlation(df):
    """
    Show a heatmap of correlations between features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print("\n" + "=" * 60)
    print("PLOTTING CORRELATION HEATMAP")
    print("=" * 60)

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) == 0:
        print("No numeric columns available for correlation heatmap")
        return

    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(min(12, len(numerical_cols)*1.2), min(10, len(numerical_cols)*1.2)))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Correlation heatmap saved as 'correlation_heatmap.png'")


def plot_target_distribution(df):
    """
    Show how many people have and do not have the disease.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print("\n" + "=" * 60)
    print("PLOTTING TARGET DISTRIBUTION")
    print("=" * 60)

    if 'cardio' not in df.columns:
        print("Column 'cardio' is missing; skipping target distribution plot")
        return

    class_counts = df['cardio'].value_counts().sort_index()
    labels = ['No Cardiovascular Disease (0)', 'Cardiovascular Disease (1)']
    values = [class_counts.get(0, 0), class_counts.get(1, 0)]

    plt.figure(figsize=(10, 6))
    class_colors = ['#2ecc71', '#e74c3c']
    bars = plt.bar(labels, values, color=class_colors, edgecolor='black', alpha=0.8)

    for bar in bars:
        bar_height = bar.get_height()
        percentage = (bar_height / len(df)) * 100 if len(df) > 0 else 0
        plt.text(bar.get_x() + bar.get_width() / 2., bar_height,
                 f'{int(bar_height)}\n({percentage:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')

    plt.title('Target Variable Distribution (Cardiovascular Disease)', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    print("Target distribution plot saved as 'target_distribution.png'")

    print(f"\nTarget class distribution:")
    print(f"  - No Disease (0): {values[0]} ({(values[0]/len(df)*100 if len(df) > 0 else 0):.1f}%)")
    print(f"  - Disease (1): {values[1]} ({(values[1]/len(df)*100 if len(df) > 0 else 0):.1f}%)")
    plt.close()


def run_all_visualizations(df):
    """
    Execute all EDA visualizations in sequence.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset
    """
    plot_distributions(df)
    plot_correlation(df)
    plot_target_distribution(df)
