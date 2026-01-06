"""
Model training module for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=100, random_state=42):
    """Train Gradient Boosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression classifier."""
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filepath):
    """Save trained model to disk."""
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load trained model from disk."""
    return joblib.load(filepath)


if __name__ == "__main__":
    # Example usage
    from preprocessing import load_data, preprocess_data
    
    # Load and preprocess data
    # df = load_data('data/raw/customer_data.csv')
    # X, y = preprocess_data(df)
    
    # Split data
    # X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    # model = train_random_forest(X_train, y_train)
    
    # Save model
    # save_model(model, 'models/churn_model.pkl')
    
    print("Training module ready")
