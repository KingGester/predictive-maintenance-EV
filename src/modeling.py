import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    # Load data
    df = pd.read_csv(data_path)
    
    # Encode target variable
    le = LabelEncoder()
    df['fault_type'] = le.fit_transform(df['fault_type'])
    
    # Split features and target
    X = df.drop('fault_type', axis=1)
    y = df['fault_type']
    
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, le

def feature_selection(X_train, y_train):
    # Use a model to select important features
    selector = SelectFromModel(
        GradientBoostingClassifier(random_state=42), threshold='median'
    )
    X_train_selected = selector.fit_transform(X_train, y_train)
    return X_train_selected, selector

def train_random_forest(X_train, y_train):
    # Hyperparameter tuning with RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, 
        n_iter=20, cv=5, random_state=42, 
        n_jobs=-1, scoring='f1_macro'
    )
    
    random_search.fit(X_train, y_train)
    best_rf = random_search.best_estimator_
    
    return best_rf

def train_xgboost(X_train, y_train):
    # Hyperparameter tuning for XGBoost
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'min_child_weight': [1, 3, 5, 7]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    random_search = RandomizedSearchCV(
        xgb, param_distributions=param_dist, 
        n_iter=20, cv=5, random_state=42, 
        n_jobs=-1, scoring='f1_macro'
    )
    
    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_
    
    return best_xgb

def stacked_ensemble(models, X_train, y_train, X_test, y_test):
    """
    Create a stacked ensemble from the given models
    """
    # Train base models and create meta-features
    meta_features_train = np.zeros((X_train.shape[0], len(models)))
    meta_features_test = np.zeros((X_test.shape[0], len(models)))
    
    for i, model in enumerate(models):
        # Cross-validation predictions for training meta-features
        cv_preds = cross_val_score(model, X_train, y_train, cv=5, method='predict_proba')
        meta_features_train[:, i] = cv_preds
        
        # Fit on full training data and predict test
        model.fit(X_train, y_train)
        meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]
    
    # Train meta-model
    meta_model = LogisticRegression()
    meta_model.fit(meta_features_train, y_train)
    
    # Predict
    final_predictions = meta_model.predict(meta_features_test)
    
    return final_predictions

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate model and print metrics
    """
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 10 Feature Importance:")
        for i in range(min(10, len(indices))):
            print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
    
    return y_pred

if __name__ == "__main__":
    data_path = "data/Fault_nev_dataset.csv"
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(data_path)
    
    # Feature selection
    X_train_selected, selector = feature_selection(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Train models
    print("Training Random Forest...")
    best_rf = train_random_forest(X_train_selected, y_train)
    
    print("Training XGBoost...")
    best_xgb = train_xgboost(X_train_selected, y_train)
    
    # Evaluate models
    print("\nRandom Forest Results:")
    evaluate_model(best_rf, X_test_selected, y_test, label_encoder)
    
    print("\nXGBoost Results:")
    evaluate_model(best_xgb, X_test_selected, y_test, label_encoder)