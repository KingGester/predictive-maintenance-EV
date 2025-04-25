import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('results', exist_ok=True)

print("Loading data...")
df = pd.read_csv("data/Fault_nev_dataset.csv")

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nClass distribution:")
print(df['fault_type'].value_counts())

print("\nData types:")
print(df.dtypes)

le = LabelEncoder()
df['fault_type'] = le.fit_transform(df['fault_type'])
fault_type_names = le.classes_
print("\nEncoded classes:", list(zip(range(len(fault_type_names)), fault_type_names)))

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", categorical_cols)

for col in categorical_cols:
    if col != 'fault_type':
        le_feat = LabelEncoder()
        df[col] = le_feat.fit_transform(df[col])
        print(f"Encoded {col}: {list(zip(range(len(le_feat.classes_)), le_feat.classes_))}")

X = df.drop('fault_type', axis=1)
y = df['fault_type']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nApplying feature engineering...")
def create_features(X):
    X_new = X.copy()
    
    numeric_cols = X_new.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        X_new[f'{col}_squared'] = X_new[col] ** 2
    
    important_cols = ['battery_voltage', 'battery_current', 'engine_temperature', 
                       'motor_efficiency', 'speed', 'acceleration']
    important_cols = [col for col in important_cols if col in numeric_cols]
    
    for i, col1 in enumerate(important_cols):
        for col2 in important_cols[i+1:]:
            X_new[f'{col1}_{col2}_interaction'] = X_new[col1] * X_new[col2]
    
    return X_new

X_train_engineered = create_features(X_train)
X_test_engineered = create_features(X_test)

print(f"Features increased from {X_train.shape[1]} to {X_train_engineered.shape[1]}")

print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_engineered)
X_test_scaled = scaler.transform(X_test_engineered)

print("\nPerforming feature selection...")
selector = SelectFromModel(
    GradientBoostingClassifier(random_state=42, n_estimators=100),
    threshold='median'
)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print(f"Selected {X_train_selected.shape[1]} out of {X_train_scaled.shape[1]} features")

selected_indices = selector.get_support(indices=True)
selected_feature_names = X_train_engineered.columns[selected_indices].tolist()
print("\nTop 10 selected features:")
for i, feature in enumerate(selected_feature_names[:10]):
    print(f"{i+1}. {feature}")

print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

print("\nTraining models...")

print("\n1. Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf.predict(X_test_selected)

print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf, target_names=fault_type_names))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=fault_type_names, yticklabels=fault_type_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.savefig('results/rf_confusion_matrix.png')
plt.close()

print("\n2. Training Gradient Boosting with hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)
grid_search = RandomizedSearchCV(
    gb, param_distributions=param_grid, 
    n_iter=10, cv=3, verbose=1, 
    n_jobs=-1, scoring='f1_weighted',
    random_state=42
)

grid_search.fit(X_train_balanced, y_train_balanced)
best_gb = grid_search.best_estimator_

print("\nBest GB Parameters:", grid_search.best_params_)
y_pred_gb = best_gb.predict(X_test_selected)

print("\nGradient Boosting Results:")
print(classification_report(y_test, y_pred_gb, target_names=fault_type_names))

feature_importance = best_gb.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
top_n = 15

plt.figure(figsize=(12, 8))
plt.barh(range(top_n), feature_importance[sorted_idx][:top_n], align='center')
plt.yticks(range(top_n), [selected_feature_names[i] for i in sorted_idx[:top_n]])
plt.title('Top 15 Feature Importance - Gradient Boosting')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('results/gb_feature_importance.png')
plt.close()

print("\n3. Creating Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', best_gb)
    ],
    voting='soft'
)

ensemble.fit(X_train_balanced, y_train_balanced)
y_pred_ensemble = ensemble.predict(X_test_selected)

print("\nEnsemble Results:")
print(classification_report(y_test, y_pred_ensemble, target_names=fault_type_names))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_ensemble)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=fault_type_names, yticklabels=fault_type_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Ensemble Confusion Matrix')
plt.tight_layout()
plt.savefig('results/ensemble_confusion_matrix.png')
plt.close()

misclassified_indices = np.where(y_test != y_pred_ensemble)[0]
misclassified_data = X_test.iloc[misclassified_indices]
misclassified_true = y_test.iloc[misclassified_indices]
misclassified_pred = y_pred_ensemble[misclassified_indices]

misclassification_df = pd.DataFrame({
    'True': [fault_type_names[i] for i in misclassified_true],
    'Predicted': [fault_type_names[i] for i in misclassified_pred]
})

print("\nTop misclassification patterns:")
print(misclassification_df.groupby(['True', 'Predicted']).size().sort_values(ascending=False).head(5))

print("\nSaving improved model and results...")
import joblib
joblib.dump(ensemble, 'results/ensemble_model.pkl')
joblib.dump(scaler, 'results/scaler.pkl')
joblib.dump(selector, 'results/feature_selector.pkl')
joblib.dump(le, 'results/label_encoder.pkl')

print("\nTraining completed successfully!")
print("Model and results saved to the 'results' directory.") 