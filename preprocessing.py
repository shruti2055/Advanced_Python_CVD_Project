"""Preprocessing module.

Prepare the cleaned dataset for model training.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(df):
    """Convert age, scale features, and split into train and test sets."""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    data_frame = df.copy()

    print("\nConverting age from days to years")
    data_frame['age'] = data_frame['age'] // 365
    print(f"Age range: {data_frame['age'].min()} to {data_frame['age'].max()} years")

    if 'id' in data_frame.columns:
        data_frame = data_frame.drop('id', axis=1)

    if 'cardio' not in data_frame.columns:
        raise KeyError("Target column 'cardio' is missing from the dataset")

    feature_data = data_frame.drop('cardio', axis=1)
    target_data = data_frame['cardio']

    print(f"Feature matrix shape: {feature_data.shape}")
    print(f"Target vector shape: {target_data.shape}")

    numeric_columns = feature_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Numeric columns to scale: {numeric_columns}")

    scaler = StandardScaler()
    feature_data_scaled = feature_data.copy()
    feature_data_scaled[numeric_columns] = scaler.fit_transform(feature_data[numeric_columns])

    print("Scaling completed")

    X_train, X_test, y_train, y_test = train_test_split(
        feature_data_scaled,
        target_data,
        test_size=0.2,
        random_state=42,
        stratify=target_data
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, feature_data_scaled.columns.tolist()
