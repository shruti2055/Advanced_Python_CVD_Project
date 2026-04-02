"""
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
    
    # Key features for initial distribution analysis
    feature_list = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    
    # Make a grid of subplots
    figure, axes = plt.subplots(2, 3, figsize=(15, 10))
    figure.suptitle('Distribution of Key Features', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, feature_name in enumerate(feature_list):
        axes[idx].hist(df[feature_name], bins=50, color='skyblue', 
                       edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {feature_name.upper()}', 
                           fontweight='bold')
        axes[idx].set_xlabel(feature_name)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(axis='y', alpha=0.3)
    
    # Hide the empty subplot
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('distribution_plots.png', dpi=300, bbox_inches='tight')
    print("Distribution plots saved as 'distribution_plots.png'")
    plt.show()


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
    
    # Get all number columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Calculate correlation between all features
    corr_matrix = df[numerical_cols].corr()
    
    # Draw the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Numerical Features', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Correlation heatmap saved as 'correlation_heatmap.png'")
    plt.show()


def plot_target_distribution(df):
    """
    Show how many people have and don't have the disease.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print("\n" + "=" * 60)
    print("PLOTTING TARGET DISTRIBUTION")
    print("=" * 60)
    
    # Count the classes
    plt.figure(figsize=(10, 6))
    class_counts = df['cardio'].value_counts()
    class_colors = ['#2ecc71', '#e74c3c']  # Green for 0, Red for 1
    
    bars = plt.bar(
        ['No Cardiovascular Disease (0)', 'Cardiovascular Disease (1)'], 
        class_counts.values, color=class_colors, edgecolor='black', alpha=0.8
    )
    
    # Add labels on the bars
    for bar in bars:
        bar_height = bar.get_height()
        percentage = (bar_height / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width()/2., bar_height,
                f'{int(bar_height)}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('Target Variable Distribution (Cardiovascular Disease)', 
             fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    print("Target distribution plot saved as 'target_distribution.png'")
    print(f"\nTarget class distribution:")
    print(f"  - No Disease (0): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)")
    print(f"  - Disease (1): {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")
    plt.show()


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
