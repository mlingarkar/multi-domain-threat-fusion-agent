"""
Data Preprocessing Module

This module prepares the synthetic multi-domain mission dataset for
machine learning by handling timestamps, missing values, categorical
encoding, and feature selection.
"""

from typing import Tuple, List
import pandas as pd


CATEGORICAL_COLUMNS = ["mission_phase", "region_type", "anomaly_type"]

DROP_COLUMNS = ["event_id", "timestamp"]


def load_data(file_path: str = "data/synthetic_multi_domain_events.csv") -> pd.DataFrame:
    """
    Load the synthetic multi-domain dataset.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by converting timestamps and handling missing values.

    Args:
        df: Raw dataset.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_columns = df.select_dtypes(include=["number"]).columns
    categorical_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features from the timestamp column.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with additional time features.
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_night_operation"] = df["hour"].apply(lambda hour: 1 if hour < 6 or hour >= 20 else 0)

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features.

    Args:
        df: DataFrame with categorical columns.

    Returns:
        Encoded DataFrame.
    """
    df = df.copy()

    columns_to_encode = [
        column for column in CATEGORICAL_COLUMNS
        if column in df.columns
    ]

    return pd.get_dummies(df, columns=columns_to_encode, drop_first=True)


def prepare_features(
    df: pd.DataFrame,
    target_column: str = "anomaly_label"
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare model features and target labels.

    Args:
        df: Cleaned and encoded DataFrame.
        target_column: Target label column.

    Returns:
        X: Feature DataFrame.
        y: Target Series.
        feature_names: List of feature names.
    """
    df = df.copy()

    columns_to_drop = [
        column for column in DROP_COLUMNS
        if column in df.columns
    ]

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    y = df[target_column]
    X = df.drop(columns=columns_to_drop + [target_column], errors="ignore")

    feature_names = list(X.columns)

    return X, y, feature_names


def preprocess_pipeline(
    file_path: str = "data/synthetic_multi_domain_events.csv"
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Run the full preprocessing pipeline.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        X: Processed feature DataFrame.
        y: Target Series.
        feature_names: List of feature names.
        cleaned_df: Cleaned DataFrame before encoding.
    """
    raw_df = load_data(file_path)
    cleaned_df = clean_data(raw_df)
    cleaned_df = add_time_features(cleaned_df)

    encoded_df = encode_categorical_features(cleaned_df)
    X, y, feature_names = prepare_features(encoded_df)

    return X, y, feature_names, cleaned_df


if __name__ == "__main__":
    X, y, feature_names, cleaned_df = preprocess_pipeline()

    print("Preprocessing completed successfully.")
    print(f"Rows: {len(cleaned_df)}")
    print(f"Features prepared: {len(feature_names)}")
    print(f"Target distribution:")
    print(y.value_counts())