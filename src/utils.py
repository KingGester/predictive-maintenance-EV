import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    auc, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
import joblib
import os
import json
from datetime import datetime

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", figsize=(10, 8)):
    """
    Plot confusion matrix with percentages
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.round(cm_norm * 100, 2)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()

def plot_roc_curves(y_true, y_probs, class_names, figsize=(10, 8)):
    """
    Plot ROC curves for multi-class classification
    """
    n_classes = len(class_names)
    
    plt.figure(figsize=figsize)
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    
    return plt.gcf()

def plot_precision_recall_curves(y_true, y_probs, class_names, figsize=(10, 8)):
    """
    Plot precision-recall curves for multi-class classification
    """
    n_classes = len(class_names)
    
    plt.figure(figsize=figsize)
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AUC = {pr_auc:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    
    return plt.gcf()

def plot_feature_importance(model, feature_names, top_n=20, figsize=(12, 10)):
    """
    Plot feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Take only top N
        indices = indices[:min(top_n, len(indices))]
        
        plt.figure(figsize=figsize)
        plt.title('Feature Importance')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        
        return plt.gcf()
    else:
        print("Model doesn't have feature_importances_ attribute")
        return None

def plot_permutation_importance(model, X, y, feature_names, top_n=20, figsize=(12, 10), random_state=42):
    """
    Plot permutation importance (more reliable than built-in feature importance)
    """
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=random_state)
    
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    sorted_idx = sorted_idx[:min(top_n, len(sorted_idx))]
    
    plt.figure(figsize=figsize)
    plt.boxplot(perm_importance.importances[sorted_idx].T, 
                vert=False, labels=[feature_names[i] for i in sorted_idx])
    plt.title("Permutation Importance")
    plt.tight_layout()
    
    return plt.gcf(), perm_importance

def plot_learning_curve(estimator, X, y, title="Learning Curve", ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), figsize=(10, 6)):
    """
    Plot learning curve to detect overfitting/underfitting
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1_weighted')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation score")
    plt.legend(loc="best")
    
    return plt.gcf()

def save_model(model, model_path, metadata=None):
    """
    Save model and metadata
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata if provided
    if metadata:
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        
        # Add timestamp
        metadata['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    print(f"Model saved to {model_path}")
    
    return model_path

def load_model(model_path):
    """
    Load model and return metadata if exists
    """
    # Load model
    model = joblib.load(model_path)
    
    # Check for metadata
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    metadata = None
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata

def get_model_metrics(y_true, y_pred, y_proba=None):
    """
    Get comprehensive model metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted')
    }
    
    # Add class-specific metrics
    class_report = classification_report(y_true, y_pred, output_dict=True)
    for class_name, class_metrics in class_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics[f'f1_class_{class_name}'] = class_metrics['f1-score']
            metrics[f'precision_class_{class_name}'] = class_metrics['precision']
            metrics[f'recall_class_{class_name}'] = class_metrics['recall']
    
    # Add ROC AUC if probabilities are provided
    if y_proba is not None:
        try:
            # For binary classification
            if y_proba.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            # For multi-class
            else:
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                )
        except:
            # ROC AUC might fail for some cases
            pass
    
    return metrics

def identify_misclassifications(y_true, y_pred, X, feature_names, class_names):
    """
    Identify and analyze misclassifications
    """
    # Convert to DataFrame if X is numpy array
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
    
    # Add true and predicted labels
    X_df['true_label'] = [class_names[i] for i in y_true]
    X_df['predicted_label'] = [class_names[i] for i in y_pred]
    X_df['correct_prediction'] = X_df['true_label'] == X_df['predicted_label']
    
    # Get misclassifications
    misclassified = X_df[~X_df['correct_prediction']]
    
    # Group by true/predicted label combinations
    error_summary = misclassified.groupby(['true_label', 'predicted_label']).size().reset_index()
    error_summary.columns = ['True Label', 'Predicted Label', 'Count']
    error_summary = error_summary.sort_values('Count', ascending=False)
    
    return misclassified, error_summary

def get_confidence_intervals(model, X, y, n_bootstraps=1000, confidence_level=0.95, random_state=42):
    """
    Calculate confidence intervals for model performance using bootstrapping
    """
    np.random.seed(random_state)
    n_samples = len(y)
    
    # Store bootstrap metrics
    bootstrap_scores = []
    
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        
        # Use sampled indices for validation
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        
        # Predict and calculate scores
        y_pred = model.predict(X_bootstrap)
        
        # Calculate f1 score for this bootstrap sample
        score = f1_score(y_bootstrap, y_pred, average='weighted')
        bootstrap_scores.append(score)
    
    # Calculate confidence intervals
    sorted_scores = np.sort(bootstrap_scores)
    lower_index = int((1 - confidence_level) / 2 * n_bootstraps)
    upper_index = int((1 + confidence_level) / 2 * n_bootstraps)
    
    lower_bound = sorted_scores[lower_index]
    upper_bound = sorted_scores[upper_index]
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'point_estimate': np.mean(bootstrap_scores),
        'std_dev': np.std(bootstrap_scores)
    }