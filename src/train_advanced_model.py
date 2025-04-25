import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import lightgbm as lgbm
from sklearn.neural_network import MLPClassifier

# Import custom modules
from preprocessing import preprocess_data, create_features, balance_classes
from modeling import train_random_forest, train_xgboost
from utils import (
    plot_confusion_matrix, plot_roc_curves, plot_feature_importance,
    save_model, get_model_metrics, identify_misclassifications, 
    plot_learning_curve, get_confidence_intervals
)

def train_models(data_path, models_to_train=None, save_path="models"):
    """
    Train multiple models and return their evaluation metrics
    """
    print("Loading and preprocessing data...")
    # Load data
    df = pd.read_csv(data_path)
    
    # Encode target
    le = LabelEncoder()
    df['fault_type'] = le.fit_transform(df['fault_type'])
    fault_types = le.classes_
    
    # Split data
    X = df.drop('fault_type', axis=1)
    y = df['fault_type']
    
    # Get feature names before potential transformation
    feature_names = X.columns.tolist()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature engineering
    print("Applying feature engineering...")
    X_train = create_features(X_train.copy())
    X_test = create_features(X_test.copy())
    
    # Update feature names after engineering
    feature_names = X_train.columns.tolist()
    
    # Balance classes
    print("Balancing classes...")
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train, method='smote')
    
    # Available models
    model_factories = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500, 
            max_depth=20, 
            min_samples_split=5, 
            min_samples_leaf=2, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            min_samples_split=5, 
            random_state=42
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            gamma=0.1, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='mlogloss'
        ),
        "catboost": lambda: CatBoostClassifier(
            iterations=200, 
            depth=6, 
            learning_rate=0.1, 
            loss_function='MultiClass', 
            random_seed=42, 
            verbose=0
        ),
        "lightgbm": lambda: lgbm.LGBMClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            num_leaves=31, 
            random_state=42
        ),
        "svm": lambda: SVC(
            C=10, 
            gamma='scale', 
            probability=True, 
            class_weight='balanced', 
            random_state=42
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(100, 50), 
            max_iter=500, 
            alpha=0.0001, 
            learning_rate='adaptive', 
            early_stopping=True, 
            random_state=42
        )
    }
    
    # Default to all models if none specified
    if models_to_train is None:
        models_to_train = list(model_factories.keys())
    
    results = {}
    trained_models = {}
    
    # Train and evaluate each model
    for model_name in models_to_train:
        if model_name not in model_factories:
            print(f"Model {model_name} not found, skipping...")
            continue
        
        print(f"\nTraining {model_name}...")
        model = model_factories[model_name]()
        
        # Train model
        model.fit(X_train_balanced, y_train_balanced)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None
        
        # Get metrics
        metrics = get_model_metrics(y_test, y_pred, y_prob)
        
        # Print results
        print(f"{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create visualizations
        os.makedirs(f"{save_path}/plots", exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        conf_matrix = plot_confusion_matrix(y_test, y_pred, fault_types)
        plt.savefig(f"{save_path}/plots/{model_name}_confusion_matrix.png")
        plt.close()
        
        # Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 10))
            feat_imp = plot_feature_importance(model, feature_names)
            plt.savefig(f"{save_path}/plots/{model_name}_feature_importance.png")
            plt.close()
        
        # Learning Curve
        plt.figure(figsize=(10, 6))
        learn_curve = plot_learning_curve(model, X_train_balanced, y_train_balanced)
        plt.savefig(f"{save_path}/plots/{model_name}_learning_curve.png")
        plt.close()
        
        # Save model
        model_metadata = {
            'model_name': model_name,
            'metrics': metrics,
            'feature_names': feature_names,
            'target_classes': fault_types.tolist(),
            'training_size': len(X_train),
            'test_size': len(X_test)
        }
        
        save_model(model, f"{save_path}/{model_name}.pkl", model_metadata)
        
        # Store results and model
        results[model_name] = metrics
        trained_models[model_name] = model
    
    # Create ensemble model if we have multiple models
    if len(trained_models) > 1:
        print("\nTraining Ensemble Model...")
        
        # Only include models that support predict_proba
        voting_estimators = []
        for name, model in trained_models.items():
            if hasattr(model, 'predict_proba'):
                voting_estimators.append((name, model))
        
        if len(voting_estimators) >= 2:
            ensemble = VotingClassifier(
                estimators=voting_estimators,
                voting='soft'
            )
            
            # Train ensemble
            ensemble.fit(X_train_balanced, y_train_balanced)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble.predict(X_test)
            y_prob_ensemble = ensemble.predict_proba(X_test)
            
            # Get metrics
            metrics_ensemble = get_model_metrics(y_test, y_pred_ensemble, y_prob_ensemble)
            
            # Print results
            print("Ensemble Results:")
            print(f"Accuracy: {metrics_ensemble['accuracy']:.4f}")
            print(f"F1 Weighted: {metrics_ensemble['f1_weighted']:.4f}")
            print(f"F1 Macro: {metrics_ensemble['f1_macro']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_ensemble))
            
            # Confusion Matrix for Ensemble
            plt.figure(figsize=(10, 8))
            conf_matrix = plot_confusion_matrix(y_test, y_pred_ensemble, fault_types)
            plt.savefig(f"{save_path}/plots/ensemble_confusion_matrix.png")
            plt.close()
            
            # Save ensemble model
            ensemble_metadata = {
                'model_name': 'ensemble',
                'base_models': [e[0] for e in voting_estimators],
                'metrics': metrics_ensemble,
                'feature_names': feature_names,
                'target_classes': fault_types.tolist(),
                'training_size': len(X_train),
                'test_size': len(X_test)
            }
            
            save_model(ensemble, f"{save_path}/ensemble.pkl", ensemble_metadata)
            
            # Add ensemble to results
            results['ensemble'] = metrics_ensemble
            trained_models['ensemble'] = ensemble
    
    # Find best model
    best_model_name = max(results, key=lambda k: results[k]['f1_weighted'])
    best_model = trained_models[best_model_name]
    
    print(f"\nBest model: {best_model_name} with F1 score: {results[best_model_name]['f1_weighted']:.4f}")
    
    # Calculate confidence intervals for best model
    conf_intervals = get_confidence_intervals(best_model, X_test, y_test)
    print(f"Confidence Interval (95%): [{conf_intervals['lower_bound']:.4f}, {conf_intervals['upper_bound']:.4f}]")
    
    # Analyze misclassifications
    misclassified, error_summary = identify_misclassifications(
        y_test, best_model.predict(X_test), X_test, feature_names, fault_types
    )
    
    print("\nTop misclassification patterns:")
    print(error_summary.head())
    
    # Save confidence intervals and error summary
    error_summary.to_csv(f"{save_path}/error_analysis.csv", index=False)
    
    return trained_models, results, best_model_name

if __name__ == "__main__":
    data_path = "data/Fault_nev_dataset.csv"
    models_dir = "models"
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Train all available models
    models, results, best_model = train_models(data_path, save_path=models_dir)
    
    print("\nTraining completed successfully!")
    print(f"Models saved to {models_dir}/")
    print(f"Best model: {best_model}") 