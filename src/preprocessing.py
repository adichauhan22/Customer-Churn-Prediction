"""
Data preprocessing module for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Fill numerical columns with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def encode_categorical_features(df, categorical_columns):
    """Encode categorical features using LabelEncoder."""
    le = LabelEncoder()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    return df


def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(df, target_column='Churn'):
    """Complete preprocessing pipeline."""
    # Handle missing values
    df = handle_missing_values(df)
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return X, y


def save_processed_data(df, filepath):
    """Save processed data to CSV file."""
    df.to_csv(filepath, index=False)
