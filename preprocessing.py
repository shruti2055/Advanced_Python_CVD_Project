"""
"""Preprocessing Module.

Prepares data for machine learning models.
Converts age and scales all features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """
    Convert age to years and scale all features.
    
    Prepares the data for model training by converting age from days to years
    and scaling all numerical features so they have mean 0 and std 1.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with age in days
        
    Returns
    -------
    tuple
        (X_processed, y) - scaled features and target variable
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Make a copy
    df_ready = df.copy()
    
    # Convert age from days to years
    print("\n--- Converting Age Units ---")
    df_ready['age'] = df_ready['age'] // 365
    print(f"Age range: {df_ready['age'].min()} - {df_ready['age'].max()} years")
    print(f"Mean age: {df_ready['age'].mean():.1f} years")
    
    # Split features and target
    print("\n--- Feature-Target Separation ---")
    # Drop 'id' (not useful for predictions)
    feature_matrix = df_ready.drop(['cardio', 'id'], axis=1)
    target_vector = df_ready['cardio']
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Target vector shape: {target_vector.shape}")
    print(f"\nAvailable features: {list(feature_matrix.columns)}")
    
    # Show types
    print("\n--- Feature Data Types ---")
    print(feature_matrix.dtypes)
    
    # Get number columns to scale
    num_feature_cols = feature_matrix.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    print(f"\nNumber columns to scale: {num_feature_cols}")
    
    # Scale all number columns
    print("\n--- Scaling Numerical Features ---")
    scaler = StandardScaler()
    feature_matrix_scaled = feature_matrix.copy()
    feature_matrix_scaled[num_feature_cols] = scaler.fit_transform(
        feature_matrix[num_feature_cols]
    )
    
    print("Features scaled successfully")
    print(f"\nScaled features check:")
    scaled_means = feature_matrix_scaled[num_feature_cols].mean().values
    scaled_stds = feature_matrix_scaled[num_feature_cols].std().values
    print(f"  Average (should be ~0): {scaled_means}")
    print(f"  Standard Dev (should be ~1): {scaled_stds}")
    
    # Done
    print(f"\n--- Preprocessing Done ---"}
    print(f"Input: {feature_matrix.shape}")
    print(f"Output: {feature_matrix_scaled.shape}")
    print(f"Target: {target_vector.shape}")
    print(f"Data ready for models")
    
    return feature_matrix_scaled, target_vector
