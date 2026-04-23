"""Data manager module.

Load the raw dataset and prepare it for modeling.
"""

import pandas as pd


def load_data(filepath):
    """Load the dataset from a CSV file."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    try:
        data_frame = pd.read_csv(filepath, sep=';')
    except Exception as error:
        print(f"Could not load data from {filepath}: {error}")
        raise

    print("Dataset loaded successfully")
    print(f"Shape: {data_frame.shape}")
    print("First five rows:")
    print(data_frame.head())
    print("Data types:")
    print(data_frame.dtypes)

    return data_frame


def clean_data(df):
    """Remove bad records and prepare the dataset for analysis."""
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    cleaned_df = df.copy()
    initial_row_count = cleaned_df.shape[0]
    print(f"Initial data shape: {cleaned_df.shape}")

    print("\nChecking for duplicate rows")
    duplicate_count = cleaned_df.duplicated().sum()
    print(f"Duplicates found: {duplicate_count}")
    if duplicate_count > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Rows after removing duplicates: {cleaned_df.shape[0]}")

    print("\nChecking for missing values")
    missing_report = cleaned_df.isnull().sum()
    if missing_report.sum() == 0:
        print("No missing values were found")
    else:
        print(missing_report[missing_report > 0])
        cleaned_df = cleaned_df.dropna()
        print(f"Rows after dropping missing values: {cleaned_df.shape[0]}")

    print("\nFiltering invalid measurements")
    if 'ap_hi' in cleaned_df.columns and 'ap_lo' in cleaned_df.columns:
        before_bp_filter = cleaned_df.shape[0]
        cleaned_df = cleaned_df[(cleaned_df['ap_hi'] > 0) & (cleaned_df['ap_lo'] > 0)]
        cleaned_df = cleaned_df[(cleaned_df['ap_hi'] <= 300) & (cleaned_df['ap_lo'] <= 300)]
        removed_bp = before_bp_filter - cleaned_df.shape[0]
        if removed_bp > 0:
            print(f"Removed {removed_bp} rows with invalid blood pressure values")
    else:
        print("Blood pressure columns not found, skipping blood pressure filtering")

    if 'height' in cleaned_df.columns and 'weight' in cleaned_df.columns:
        cleaned_df = cleaned_df[(cleaned_df['height'] > 0) & (cleaned_df['weight'] > 0)]
        print("Removed rows with non-positive height or weight")
    else:
        print("Height or weight columns not found, skipping height and weight filtering")

    print("\nCleaning summary")
    print(f"Final shape after cleaning: {cleaned_df.shape}")
    removed_rows = initial_row_count - cleaned_df.shape[0]
    print(f"Total rows removed: {removed_rows}")

    return cleaned_df
