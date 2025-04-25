import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def load_data(filepath):
    return pd.read_csv(filepath)

def encode_categorical_features(df, categorical_cols, method='label'):
    df_encoded = df.copy()
    
    if method == 'label':
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
        return df_encoded, label_encoders
    
    elif method == 'onehot':
        return pd.get_dummies(df_encoded, columns=categorical_cols)
    
    else:
        raise ValueError("Method must be 'label' or 'onehot'")

def handle_missing_values(df, numeric_cols, categorical_cols, numeric_strategy='mean', categorical_strategy='most_frequent'):
    df_processed = df.copy()
    
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        df_processed[numeric_cols] = numeric_imputer.fit_transform(df_processed[numeric_cols])
    
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        df_processed[categorical_cols] = categorical_imputer.fit_transform(df_processed[categorical_cols])
    
    return df_processed

def normalize_features(df, numeric_cols, method='standard'):
    df_normalized = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
    
    return df_normalized, scaler

def split_data(df, target_col, test_size=0.2, random_state=42, stratify=True):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    return X_train, X_test, y_train, y_test

def handle_class_imbalance(X, y, method='smote', sampling_strategy='auto', random_state=42):
    if method == 'smote':
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == 'adasyn':
        sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == 'undersample':
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        raise ValueError("Method must be 'smote', 'adasyn', or 'undersample'")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    return X_resampled, y_resampled

def feature_selection(X, y, method='kbest', n_features=10, random_state=42):
    if method == 'kbest':
        selector = SelectKBest(f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        return X_selected, selector
    
    elif method == 'rfe':
        estimator = RandomForestClassifier(random_state=random_state)
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        return X_selected, selector
    
    elif method == 'model_based':
        model = RandomForestClassifier(random_state=random_state)
        selector = SelectFromModel(model, max_features=n_features)
        X_selected = selector.fit_transform(X, y)
        return X_selected, selector
    
    else:
        raise ValueError("Method must be 'kbest', 'rfe', or 'model_based'")

def dimensionality_reduction(X, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

def get_preprocessing_pipeline(numeric_features, categorical_features, 
                               scaling_method='standard', 
                               numeric_impute_strategy='mean',
                               categorical_impute_strategy='most_frequent'):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_impute_strategy)),
        ('scaler', get_scaler(scaling_method))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_impute_strategy)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def get_scaler(method):
    if method == 'standard':
        return StandardScaler()
    elif method == 'minmax':
        return MinMaxScaler()
    elif method == 'robust':
        return RobustScaler()
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")

def detect_outliers(df, columns, method='zscore', threshold=3):
    if method == 'zscore':
        from scipy import stats
        
        outlier_indices = []
        
        for col in columns:
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = np.where(z_scores > threshold)[0]
            outlier_indices.extend(outliers)
        
        outlier_indices = list(set(outlier_indices))
        return outlier_indices
    
    elif method == 'iqr':
        outlier_indices = []
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_indices.extend(outliers)
        
        outlier_indices = list(set(outlier_indices))
        return outlier_indices
    
    else:
        raise ValueError("Method must be 'zscore' or 'iqr'")

def handle_outliers(df, outlier_indices, method='remove'):
    if method == 'remove':
        return df.drop(outlier_indices)
    
    elif method == 'cap':
        df_capped = df.copy()
        
        for col in df.select_dtypes(include=np.number).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_capped[col] = np.where(df_capped[col] < lower_bound, lower_bound, df_capped[col])
            df_capped[col] = np.where(df_capped[col] > upper_bound, upper_bound, df_capped[col])
        
        return df_capped
    
    else:
        raise ValueError("Method must be 'remove' or 'cap'")

def create_feature_crosses(df, features_to_cross):
    df_with_crosses = df.copy()
    
    for feature_pair in features_to_cross:
        if len(feature_pair) != 2:
            continue
            
        feature1, feature2 = feature_pair
        
        if feature1 in df.columns and feature2 in df.columns:
            new_feature_name = f"{feature1}_{feature2}_cross"
            
            if df[feature1].dtype.kind in 'bifc' and df[feature2].dtype.kind in 'bifc':
                df_with_crosses[new_feature_name] = df[feature1] * df[feature2]
            else:
                df_with_crosses[new_feature_name] = df[feature1].astype(str) + '_' + df[feature2].astype(str)
    
    return df_with_crosses