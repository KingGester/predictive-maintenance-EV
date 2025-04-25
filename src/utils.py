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
import logging
import sys
from pathlib import Path

def setup_logging(log_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    logger = logging.getLogger('ev_predictor')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", figsize=(10, 8)):
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
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attributes")
    
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=figsize)
    plt.title("Feature Importance")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    return plt.gcf()

def plot_permutation_importance(model, X, y, feature_names, top_n=20, figsize=(12, 10), random_state=42):
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

def save_model(model, model_name, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{model_name}_{timestamp}.pkl")
    
    joblib.dump(model, model_path)
    
    return model_path

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def get_model_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted')
    }
    
    class_report = classification_report(y_true, y_pred, output_dict=True)
    for class_name, class_metrics in class_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics[f'f1_class_{class_name}'] = class_metrics['f1-score']
            metrics[f'precision_class_{class_name}'] = class_metrics['precision']
            metrics[f'recall_class_{class_name}'] = class_metrics['recall']
    
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                )
        except:
            pass
    
    return metrics

def identify_misclassifications(y_true, y_pred, X, feature_names, class_names):
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
    
    X_df['true_label'] = [class_names[i] for i in y_true]
    X_df['predicted_label'] = [class_names[i] for i in y_pred]
    X_df['correct_prediction'] = X_df['true_label'] == X_df['predicted_label']
    
    misclassified = X_df[~X_df['correct_prediction']]
    
    error_summary = misclassified.groupby(['true_label', 'predicted_label']).size().reset_index()
    error_summary.columns = ['True Label', 'Predicted Label', 'Count']
    error_summary = error_summary.sort_values('Count', ascending=False)
    
    return misclassified, error_summary

def get_confidence_intervals(model, X, y, n_bootstraps=1000, confidence_level=0.95, random_state=42):
    np.random.seed(random_state)
    n_samples = len(y)
    
    bootstrap_scores = []
    
    for i in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        
        y_pred = model.predict(X_bootstrap)
        
        score = f1_score(y_bootstrap, y_pred, average='weighted')
        bootstrap_scores.append(score)
    
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

def get_model_summary(model):
    model_type = type(model).__name__
    
    summary = {
        'model_type': model_type,
        'params': {}
    }
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
        summary['params'] = params
    
    if hasattr(model, 'feature_importances_'):
        summary['has_feature_importances'] = True
    
    if hasattr(model, 'coef_'):
        summary['has_coefficients'] = True
    
    if hasattr(model, 'n_features_in_'):
        summary['n_features'] = model.n_features_in_
    
    if hasattr(model, 'classes_'):
        summary['n_classes'] = len(model.classes_)
        summary['classes'] = model.classes_.tolist() if hasattr(model.classes_, 'tolist') else model.classes_
    
    return summary

def create_results_directory(base_dir='results'):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def save_dataframe(df, filename, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
    else:
        path = filename
    
    if filename.endswith('.csv'):
        df.to_csv(path, index=False)
    elif filename.endswith('.parquet'):
        df.to_parquet(path, index=False)
    elif filename.endswith('.pkl'):
        df.to_pickle(path)
    else:
        df.to_csv(path + '.csv', index=False)
    
    return path

def load_dataframe(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.pkl'):
        return pd.read_pickle(file_path)
    else:
        return pd.read_csv(file_path)

def plot_training_history(history, figsize=(12, 5)):
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if 'accuracy' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
        
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    return plt.gcf()

def analyze_predictions(y_true, y_pred, X_test, feature_names=None):
    errors = (y_true != y_pred)
    error_indices = np.where(errors)[0]
    
    results = {
        'total_samples': len(y_true),
        'correct_predictions': len(y_true) - len(error_indices),
        'incorrect_predictions': len(error_indices),
        'accuracy': accuracy_score(y_true, y_pred),
        'error_indices': error_indices.tolist()
    }
    
    if len(error_indices) > 0 and feature_names is not None:
        error_features = X_test[error_indices]
        
        if isinstance(X_test, pd.DataFrame):
            error_samples = X_test.iloc[error_indices]
        else:
            error_samples = pd.DataFrame(error_features, columns=feature_names)
        
        error_samples['true_label'] = y_true[error_indices]
        error_samples['predicted_label'] = y_pred[error_indices]
        
        results['error_samples'] = error_samples
    
    return results

def compare_models(model_results, metric='accuracy', figsize=(12, 8)):
    model_names = list(model_results.keys())
    metrics = [model_results[model][metric] for model in model_names]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(model_names, metrics)
    
    plt.xlabel('Models')
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison by {metric.capitalize()}')
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    return plt.gcf()

def prepare_deployment_package(model, model_path, metadata, output_dir='deployment'):
    os.makedirs(output_dir, exist_ok=True)
    
    deployed_model_path = os.path.join(output_dir, 'model.pkl')
    metadata_path = os.path.join(output_dir, 'metadata.json')
    
    joblib.dump(model, deployed_model_path)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return {
        'model_path': deployed_model_path,
        'metadata_path': metadata_path
    }

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        print(f"Function {func.__name__} took {execution_time:.2f} seconds to execute.")
        return result
    return wrapper

def generate_model_report(model, X_train, y_train, X_test, y_test, feature_names=None, output_dir=None):
    results = {}
    
    start_time = datetime.datetime.now()
    
    results['model_summary'] = get_model_summary(model)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    results['training'] = {
        'accuracy': accuracy_score(y_train, train_pred),
        'precision': precision_score(y_train, train_pred, average='weighted'),
        'recall': recall_score(y_train, train_pred, average='weighted'),
        'f1': f1_score(y_train, train_pred, average='weighted')
    }
    
    results['testing'] = {
        'accuracy': accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred, average='weighted'),
        'recall': recall_score(y_test, test_pred, average='weighted'),
        'f1': f1_score(y_test, test_pred, average='weighted')
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'model_report.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        plt.figure(figsize=(12, 10))
        plot_confusion_matrix(y_test, test_pred, title='Test Set Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'test_confusion_matrix.png'))
        
        if feature_names is not None and (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')):
            plt.figure(figsize=(12, 10))
            plot_feature_importance(model, feature_names)
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    results['execution_time'] = execution_time
    
    return results