"""
"""Data Manager Module.

Loads raw CSV data and cleans it.
Removes duplicates, handles missing values, and filters out bad records.
"""

import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Load the dataset from a CSV file and show basic info.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        The loaded dataset
    """
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    # Load CSV—dataset uses semicolon as delimiter
    raw_data = pd.read_csv(filepath, sep=';')
    
    # Print basic info
    print(f"\nDataset loaded successfully")
    print(f"Shape: {raw_data.shape[0]} rows x {raw_data.shape[1]} columns")
    
    print("\n--- Column Names and Data Types ---")
    print(raw_data.dtypes)
    
    print("\n--- First 5 Rows ---")
    print(raw_data.head())
    
    print("\n--- Dataset Info ---")
    print(raw_data.info())
    
    return raw_data


def clean_data(df):
    """
    Clean the dataset by removing bad records.
    
    Removes duplicates, handles missing values, and removes records with
    invalid data (e.g., negative blood pressure, zero height or weight).
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    """
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    
    # Make a copy so we don't change the original
    cleaned_df = df.copy()
    initial_row_count = cleaned_df.shape[0]
    
    print(f"\nInitial shape: {cleaned_df.shape}")
    
    # Remove duplicate records
    print("\n--- Checking for Duplicates ---")
    num_duplicates = cleaned_df.duplicated().sum()
    print(f"Duplicate records found: {num_duplicates}")
    if num_duplicates > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"After deduplication: {cleaned_df.shape[0]} rows")
    
    # Handle missing values
    print("\n--- Missing Values Summary ---")
    missing_count = cleaned_df.isnull().sum()
    if missing_count.sum() == 0:
        print("No missing values found")
    else:
        print(missing_count[missing_count > 0])
        cleaned_df = cleaned_df.dropna()
        print(f"Records after dropna: {cleaned_df.shape[0]}")
    
    # Filter out invalid measurements
    print("\n--- Filtering Invalid Values ---")
    
    # Blood pressure should be positive and within physiological range
    invalid_ap_hi = (cleaned_df['ap_hi'] < 0).sum()
    invalid_ap_lo = (cleaned_df['ap_lo'] < 0).sum()
    print(f"Negative systolic BP: {invalid_ap_hi}")
    print(f"Negative diastolic BP: {invalid_ap_lo}")
    
    rows_before_bp_filter = cleaned_df.shape[0]
    # Keep only positive values and exclude extreme outliers
    cleaned_df = cleaned_df[(cleaned_df['ap_hi'] > 0) & (cleaned_df['ap_lo'] > 0)]
    cleaned_df = cleaned_df[(cleaned_df['ap_hi'] <= 300) & (cleaned_df['ap_lo'] <= 300)]
    rows_removed_bp = rows_before_bp_filter - cleaned_df.shape[0]
    
    if rows_removed_bp > 0:
        print(f"Removed {rows_removed_bp} records with invalid BP")
    
    # Height and weight must be positive
    cleaned_df = cleaned_df[(cleaned_df['height'] > 0) & (cleaned_df['weight'] > 0)]
    print(f"Filtered records with invalid height/weight")
    
    # Summary
    print(f"\n--- Cleaning Summary ---")
    print(f"Final shape: {cleaned_df.shape}")
    total_rows_removed = initial_row_count - cleaned_df.shape[0]
    print(f"Total records removed: {total_rows_removed}")
    
    return cleaned_df
