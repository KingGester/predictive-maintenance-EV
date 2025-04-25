import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import joblib
import os
import time
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import plot_confusion_matrix, save_model, get_model_metrics

def train_logistic_regression(X_train, y_train, 
                             penalty='l2', 
                             C=1.0, 
                             solver='lbfgs', 
                             max_iter=1000,
                             class_weight=None,
                             random_state=42):
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, 
                       max_depth=None, 
                       min_samples_split=2, 
                       min_samples_leaf=1,
                       max_features=None,
                       criterion='gini',
                       class_weight=None,
                       random_state=42):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, 
                       n_estimators=100, 
                       max_depth=None, 
                       min_samples_split=2, 
                       min_samples_leaf=1,
                       max_features='sqrt',
                       criterion='gini',
                       class_weight=None,
                       random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, 
                           n_estimators=100, 
                           learning_rate=0.1, 
                           max_depth=3,
                           min_samples_split=2,
                           min_samples_leaf=1,
                           max_features=None,
                           subsample=1.0,
                           random_state=42):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        subsample=subsample,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, 
             C=1.0, 
             kernel='rbf', 
             gamma='scale',
             degree=3,
             class_weight=None,
             probability=True,
             random_state=42):
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train, var_smoothing=1e-9):
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, 
             n_neighbors=5, 
             weights='uniform', 
             algorithm='auto',
             leaf_size=30,
             p=2,
             metric='minkowski'):
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric
    )
    
    model.fit(X_train, y_train)
    return model

def train_mlp(X_train, y_train, 
             hidden_layer_sizes=(100,), 
             activation='relu', 
             solver='adam',
             alpha=0.0001,
             batch_size='auto',
             learning_rate='constant',
             learning_rate_init=0.001,
             max_iter=200,
             shuffle=True,
             random_state=42):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        shuffle=shuffle,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, 
                 n_estimators=100, 
                 learning_rate=0.1, 
                 max_depth=3,
                 min_child_weight=1,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 gamma=0,
                 reg_alpha=0,
                 reg_lambda=1,
                 random_state=42):
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, 
                  n_estimators=100, 
                  learning_rate=0.1, 
                  max_depth=-1,
                  num_leaves=31,
                  min_child_samples=20,
                  subsample=1.0,
                  colsample_bytree=1.0,
                  reg_alpha=0,
                  reg_lambda=0,
                  random_state=42):
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def train_catboost(X_train, y_train, 
                  iterations=100, 
                  learning_rate=0.1, 
                  depth=6,
                  l2_leaf_reg=3,
                  random_strength=1,
                  bagging_temperature=1,
                  random_state=42,
                  verbose=0):
    model = cb.CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        random_seed=random_state,
        verbose=verbose
    )
    
    model.fit(X_train, y_train)
    return model

def train_ensemble_voting(X_train, y_train, models, voting='soft'):
    estimators = [(f'model_{i}', model) for i, model in enumerate(models)]
    
    ensemble = VotingClassifier(estimators=estimators, voting=voting)
    ensemble.fit(X_train, y_train)
    
    return ensemble

def train_stacking_ensemble(X_train, y_train, 
                          base_models, 
                          meta_model=None,
                          cv=5):
    if meta_model is None:
        meta_model = LogisticRegression()
    
    estimators = [(f'model_{i}', model) for i, model in enumerate(base_models)]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=cv
    )
    
    stacking.fit(X_train, y_train)
    return stacking

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm, y_pred

def perform_grid_search(model, param_grid, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_,
        'search_time': search_time
    }
    
    return results

def perform_randomized_search(model, param_distributions, X_train, y_train, 
                             n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42):
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    results = {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'best_estimator': random_search.best_estimator_,
        'cv_results': random_search.cv_results_,
        'search_time': search_time
    }
    
    return results

def perform_cross_validation(model, X, y, cv=5, scoring='accuracy'):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    results = {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores)
    }
    
    return results

def feature_importance_analysis(model, feature_names):
    importance_values = None
    importance_type = None
    
    if hasattr(model, 'feature_importances_'):
        importance_values = model.feature_importances_
        importance_type = 'feature_importances_'
    elif hasattr(model, 'coef_'):
        importance_values = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        importance_type = 'coefficients'
    
    if importance_values is not None:
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        })
        
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        return {
            'importance_values': importance_values,
            'importance_type': importance_type,
            'feature_importance_df': feature_importance
        }
    else:
        return None

def plot_feature_importances(feature_importance_dict, top_n=20, figsize=(12, 8)):
    if feature_importance_dict is None:
        print("No feature importance data available.")
        return None
    
    feature_importance_df = feature_importance_dict['feature_importance_df']
    importance_type = feature_importance_dict['importance_type']
    
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    
    plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1])
    
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Features ({importance_type})')
    plt.tight_layout()
    
    return plt.gcf()

def get_default_models(random_state=42):
    models = {
        'logistic_regression': LogisticRegression(random_state=random_state),
        'decision_tree': DecisionTreeClassifier(random_state=random_state),
        'random_forest': RandomForestClassifier(random_state=random_state),
        'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
        'svm': SVC(probability=True, random_state=random_state),
        'naive_bayes': GaussianNB(),
        'knn': KNeighborsClassifier(),
        'mlp': MLPClassifier(random_state=random_state),
        'xgboost': xgb.XGBClassifier(random_state=random_state),
        'lightgbm': lgb.LGBMClassifier(random_state=random_state),
        'catboost': cb.CatBoostClassifier(random_seed=random_state, verbose=0)
    }
    
    return models

def get_default_param_grids():
    param_grids = {
        'logistic_regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'class_weight': [None, 'balanced']
        },
        'decision_tree': {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'balanced']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 1],
            'class_weight': [None, 'balanced']
        },
        'naive_bayes': {
            'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        },
        'mlp': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        },
        'lightgbm': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'num_leaves': [31, 50, 100],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        },
        'catboost': {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7],
            'random_strength': [0.1, 1, 10],
            'bagging_temperature': [0, 1, 10]
        }
    }
    
    return param_grids

def compare_models(X_train, y_train, X_test, y_test, models_dict=None, random_state=42):
    if models_dict is None:
        models_dict = get_default_models(random_state=random_state)
    
    results = {}
    trained_models = {}
    
    for name, model in models_dict.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Evaluating {name}...")
        metrics, cm, _ = evaluate_model(model, X_test, y_test)
        metrics['training_time'] = training_time
        
        results[name] = metrics
        trained_models[name] = model
    
    results_df = pd.DataFrame(results).T
    
    return results_df, trained_models

def find_best_model(results_df, metric='accuracy'):
    best_model_name = results_df[metric].idxmax()
    best_score = results_df.loc[best_model_name, metric]
    
    best_model_results = {
        'model_name': best_model_name,
        f'best_{metric}': best_score,
        'all_metrics': results_df.loc[best_model_name].to_dict()
    }
    
    return best_model_results

def plot_model_comparison(results_df, metric='accuracy', figsize=(12, 8)):
    plt.figure(figsize=figsize)
    
    results_sorted = results_df.sort_values(metric, ascending=False)
    
    ax = results_sorted[metric].plot(kind='bar')
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison by {metric.capitalize()}')
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(results_sorted[metric]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_model_metrics_comparison(results_df, metrics=None, figsize=(15, 10)):
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    metrics_df = results_df[metrics].copy()
    
    sorted_models = metrics_df['accuracy'].sort_values(ascending=False).index
    metrics_df = metrics_df.loc[sorted_models]
    
    plt.figure(figsize=figsize)
    
    metrics_df.plot(kind='bar', figsize=figsize)
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    return plt.gcf()

def train_and_optimize_model(model_type, X_train, y_train, X_test, y_test, 
                            param_grid=None, 
                            search_method='grid',
                            n_iter=10,
                            cv=5,
                            scoring='accuracy',
                            random_state=42):
    models = get_default_models(random_state=random_state)
    param_grids = get_default_param_grids()
    
    if model_type not in models:
        raise ValueError(f"Model type '{model_type}' not supported. Choose from: {list(models.keys())}")
    
    model = models[model_type]
    
    if param_grid is None:
        if model_type in param_grids:
            param_grid = param_grids[model_type]
        else:
            raise ValueError(f"No default param_grid for '{model_type}' and none was provided.")
    
    print(f"Optimizing {model_type} with {search_method} search...")
    
    if search_method == 'grid':
        search_results = perform_grid_search(
            model=model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            scoring=scoring
        )
    elif search_method == 'random':
        search_results = perform_randomized_search(
            model=model,
            param_distributions=param_grid,
            X_train=X_train,
            y_train=y_train,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state
        )
    else:
        raise ValueError(f"Search method '{search_method}' not supported. Choose from: ['grid', 'random']")
    
    best_model = search_results['best_estimator']
    
    print("Evaluating best model...")
    metrics, cm, y_pred = evaluate_model(best_model, X_test, y_test)
    
    results = {
        'model_type': model_type,
        'best_params': search_results['best_params'],
        'cv_score': search_results['best_score'],
        'test_metrics': metrics,
        'confusion_matrix': cm,
        'search_time': search_results['search_time'],
        'best_model': best_model,
        'y_pred': y_pred
    }
    
    return results

def create_advanced_ensemble(X_train, y_train, models_dict=None, meta_learner=None, ensemble_type='stacking', voting='soft', cv=5, random_state=42):
    if models_dict is None:
        models_dict = get_default_models(random_state=random_state)
    
    models_list = list(models_dict.values())
    
    if ensemble_type == 'voting':
        ensemble = train_ensemble_voting(X_train, y_train, models_list, voting=voting)
    elif ensemble_type == 'stacking':
        if meta_learner is None:
            meta_learner = LogisticRegression(random_state=random_state)
        ensemble = train_stacking_ensemble(X_train, y_train, models_list, meta_model=meta_learner, cv=cv)
    else:
        raise ValueError(f"Ensemble type '{ensemble_type}' not supported. Choose from: ['voting', 'stacking']")
    
    return ensemble

def evaluate_and_visualize_model(model, X_train, y_train, X_test, y_test, feature_names=None, class_names=None, figsize=(12, 10)):
    results = {}
    
    train_pred = model.predict(X_train)
    train_metrics, train_cm, _ = evaluate_model(model, X_train, y_train)
    
    test_metrics, test_cm, y_pred = evaluate_model(model, X_test, y_test)
    
    results['train_metrics'] = train_metrics
    results['test_metrics'] = test_metrics
    
    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
        if feature_names is not None:
            importance_dict = feature_importance_analysis(model, feature_names)
            results['feature_importance'] = importance_dict
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if class_names is None:
        class_names = np.unique(y_test)
    
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Train Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Test Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    results['confusion_matrices'] = fig
    
    print(f"Train Metrics: {train_metrics}")
    print(f"Test Metrics: {test_metrics}")
    
    if 'feature_importance' in results:
        feature_fig = plot_feature_importances(results['feature_importance'])
        results['feature_importance_plot'] = feature_fig
    
    return results, y_pred

def save_model_with_metadata(model, model_path, metadata=None):
    if metadata is None:
        metadata = {}
    
    metadata['model_type'] = type(model).__name__
    
    if hasattr(model, 'get_params'):
        metadata['params'] = model.get_params()
    
    metadata['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    return save_model(model, model_path, metadata)