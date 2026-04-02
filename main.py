"""Main Pipeline.

Runs the full workflow to load, clean, and explore data.
Preprocesses it for machine learning models.
"""

from data_manager import load_data, clean_data
from visualization import run_all_visualizations
from preprocessing import preprocess_data


def main():
    """
    Run the full data pipeline from start to finish.
    
    Loads data, cleans it, makes visualizations, and prepares it for models.
    
    Returns
    -------
    tuple
        (df_raw, df_clean, X_processed, y)
    """
    
    print("\n")
    print("=" * 60)
    print(" CARDIOVASCULAR DISEASE PREDICTION PROJECT ".center(60))
    print(" Milestone 3: Data Exploration & Preprocessing ".center(60))
    print("=" * 60)
    
    # Set path to data
    dataset_path = r"C:\Users\Sanu\Documents\Advanced_Python\Project\cardio_train.csv"
    
    # Step 1: Load data
    print("\n[1/5] Loading dataset...")
    df_raw = load_data(dataset_path)
    
    # Step 2: Clean data
    print("\n[2/5] Cleaning data...")
    df_clean = clean_data(df_raw)
    
    # Step 3: Make plots
    print("\n[3/5] Running EDA visualizations...")
    run_all_visualizations(df_clean)
    
    # Step 4: Prepare data
    print("\n[4/5] Preprocessing data...")
    X_processed, y = preprocess_data(df_clean)
    
    # Done
    print("\n[5/5] Pipeline complete!")
    print("\n" + "=" * 60)
    print("MILESTONE 3 SUMMARY")
    print("=" * 60)
    print("Done. Dataset loaded")
    print("Done. Data cleaned")
    print("Done. Plots created:")
    print("    - distribution_plots.png")
    print("    - correlation_heatmap.png")
    print("    - target_distribution.png")
    print("Done. Features scaled and ready")
    print(f"\nFinal shapes:")
    print(f"  - Features: {X_processed.shape}")
    print(f"  - Target: {y.shape}")
    print("\nNext: Train models (Milestone 4)")
    print("=" * 60)
    
    return df_raw, df_clean, X_processed, y


if __name__ == "__main__":
    # Run the pipeline
    df_raw, df_clean, X_processed, y = main()
