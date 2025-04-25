import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

def load_data(file_path):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(file_path)
    return df

def check_data_quality(df):
    """
    Check data quality and return a report
    """
    # Basic info
    info = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=np.number).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Check for class imbalance in target column if it exists
    if 'fault_type' in df.columns:
        info['target_distribution'] = df['fault_type'].value_counts().to_dict()
    
    # Summary statistics
    info['numeric_stats'] = df.describe().to_dict()
    
    return info

def handle_missing_values(df, strategy='knn'):
    """
    Handle missing values using different strategies
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # For numeric columns
    if strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    elif strategy == 'median':
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # For categorical columns
    if len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    
    return df

def create_features(df):
    """
    Feature engineering to create new features from existing ones
    """
    # Check if the expected columns exist in the dataset
    # Assuming columns like voltage, current, temperature exist
    
    # Create features based on domain knowledge for EV predictive maintenance
    
    # Calculate power if voltage and current exist
    if 'voltage' in df.columns and 'current' in df.columns:
        df['power'] = df['voltage'] * df['current']
    
    # Create temperature-related features if temperature exists
    if 'temperature' in df.columns:
        df['temp_squared'] = df['temperature'] ** 2
        
        # Create temperature change rate if timestamp exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df['temp_change'] = df['temperature'].diff()
            
    # Create statistical features across groups if applicable
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # If there's a vehicle ID or similar grouping column
    if 'vehicle_id' in df.columns:
        for col in numeric_cols:
            if col != 'vehicle_id':
                # Rolling statistics (window size can be adjusted)
                group = df.groupby('vehicle_id')[col]
                df[f'{col}_rolling_mean'] = group.transform(lambda x: x.rolling(min_periods=1, window=3).mean())
                df[f'{col}_rolling_std'] = group.transform(lambda x: x.rolling(min_periods=1, window=3).std())
    
    return df

def scale_features(df, scaler_type='standard'):
    """
    Scale numeric features
    """
    # Separate categorical columns if any
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Don't scale the target if it exists
    if 'fault_type' in num_cols:
        num_cols.remove('fault_type')
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df, scaler

def select_features(X, y, method='mutual_info', k=10):
    """
    Select top k features using different methods
    """
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=k)
    elif method == 'f_classif':
        selector = SelectKBest(f_classif, k=k)
    
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    
    return X_selected, selected_indices, selector

def create_pca_features(X, n_components=0.95):
    """
    Create PCA features that explain 95% of variance
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca, pca

def balance_classes(X, y, method='smote'):
    """
    Balance classes using different methods
    """
    if method == 'smote':
        balancer = SMOTE(random_state=42)
    elif method == 'adasyn':
        balancer = ADASYN(random_state=42)
    elif method == 'undersample':
        balancer = RandomUnderSampler(random_state=42)
    
    X_balanced, y_balanced = balancer.fit_resample(X, y)
    
    return X_balanced, y_balanced

def preprocess_data(data_path, engineer_features=True, handle_missing=True, 
                   feature_selection=True, balance=True):
    """
    Full preprocessing pipeline
    """
    # Load data
    df = load_data(data_path)
    
    # Check data quality
    quality_report = check_data_quality(df)
    
    # Handle missing values if needed
    if handle_missing and any(quality_report['missing_values'].values()):
        df = handle_missing_values(df)
    
    # Create new features
    if engineer_features:
        df = create_features(df)
    
    # Extract X and y
    if 'fault_type' in df.columns:
        X = df.drop('fault_type', axis=1)
        y = df['fault_type']
    else:
        # If no target column, return just the features
        return df, None, None, None
    
    # Scale features
    X_scaled, scaler = scale_features(X)
    
    # Feature selection
    if feature_selection:
        X_selected, selected_features, selector = select_features(X_scaled, y)
    else:
        X_selected = X_scaled
        selected_features = list(X.columns)
        selector = None
    
    # Balance classes
    if balance:
        X_balanced, y_balanced = balance_classes(X_selected, y)
    else:
        X_balanced, y_balanced = X_selected, y
    
    return X_balanced, y_balanced, selected_features, quality_report

if __name__ == "__main__":
    # Example usage
    data_path = "data/Fault_nev_dataset.csv"
    
    # Full preprocessing
    X, y, features, report = preprocess_data(data_path)
    
    # Print report
    print(f"Dataset shape after preprocessing: {X.shape}")
    print(f"Selected features: {features}")
    print(f"Class distribution: {np.bincount(y)}")