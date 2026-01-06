"""
Model evaluation module for customer churn prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred


def print_evaluation_metrics(metrics):
    """Print evaluation metrics."""
    print("Model Evaluation Metrics:")
    print("-" * 40)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.capitalize()}: {metric_value:.4f}")


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def print_classification_report(y_test, y_pred):
    """Print detailed classification report."""
    print("\nClassification Report:")
    print("-" * 40)
    print(classification_report(y_test, y_pred))


def get_feature_importance(model, feature_names, top_n=10):
    """Get top N important features."""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        return importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None


def plot_feature_importance(importance_df, save_path=None):
    """Plot feature importance."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Top Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    print("Evaluation module ready")
