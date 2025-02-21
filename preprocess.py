import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import pickle

# ===============================
# 1. Define Paths
# ===============================
dataset_file = r"C:\Users\nwala\Documents\Spring Final Year project 2025\Dataset.csv"
if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"Dataset not found at: {dataset_file}")
print(f"Dataset found at: {dataset_file}")

# ===============================
# 2. Load Dataset
# ===============================
print("Loading dataset...")
data = pd.read_csv(dataset_file)

# Drop the 'Unnamed: 0' and 'Patient_ID' columns if they exist
columns_to_drop = ["Unnamed: 0", "Patient_ID"]
data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

print(f"Dataset Shape: {data.shape}")
print(f"Columns: {list(data.columns)}\n")

# ===============================
# 3. Handle Missing Values
# ===============================
print("Handling missing values...")
data.fillna(data.mean(numeric_only=True), inplace=True)  # Fill numeric columns with mean
data.fillna("Unknown", inplace=True)  # Fill categorical columns with "Unknown"

print("Missing values handled.\n")

# ===============================
# 4. Separate Features and Target
# ===============================
X = data.drop(columns=["SepsisLabel"])  # Features
y = data["SepsisLabel"]  # Target variable

# ===============================
# 5. Handle Class Imbalance
# ===============================
print("Balancing the dataset...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
print(f"Resampled dataset shape: {X_resampled.shape}\n")

# Reduce dataset size for faster training
print("Reducing dataset size for faster training...")
sample_fraction = 0.2  # Use 20% of the resampled dataset
X_resampled, y_resampled = X_resampled.sample(frac=sample_fraction, random_state=42), y_resampled.sample(frac=sample_fraction, random_state=42)
print(f"Reduced dataset shape: {X_resampled.shape}\n")

# ===============================
# 6. Normalize Features
# ===============================
print("Normalizing features...")
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# ===============================
# 7. Split Data
# ===============================
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}\n")

# ===============================
# 8. Train Models
# ===============================
print("Training models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        random_state=42, n_estimators=50, max_depth=10, n_jobs=-1  # Optimize Random Forest
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", n_estimators=50, max_depth=10
    )
}

trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model

# ===============================
# 9. Evaluate Models
# ===============================
print("\nEvaluating models...")
best_model = None
best_roc_auc = 0
for name, model in trained_models.items():
    print(f"\n{name} Evaluation:")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    print(f"Accuracy: {acc:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    if roc_auc > best_roc_auc:
        best_model = model
        best_roc_auc = roc_auc

# ===============================
# 10. Save Best Model
# ===============================
print("\nSaving the best model...")
with open("sepsis_best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)
print("Best model saved as 'sepsis_best_model.pkl'.")
