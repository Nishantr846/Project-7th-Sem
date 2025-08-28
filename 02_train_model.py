# 02_train_model.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score  # <-- ADD THIS IMPORT
import os

print("--- 🚀 Stage 2: Training Model ---")

# --- Configuration ---
OUTPUT_DIR = 'outputs'
TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, 'train_data.joblib')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'symptom_checker_model.joblib')
SELECTOR_PATH = os.path.join(OUTPUT_DIR, 'feature_selector.joblib')
SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'selected_feature_names.joblib')

# Load the training data
train_data = joblib.load(TRAIN_DATA_PATH)
X_train = train_data['X_train']
y_train = train_data['y_train']
feature_names = train_data['columns']

# --- 1. Feature Selection ---
print("Performing feature selection...")
selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
selector_model.fit(X_train, y_train)

selector = SelectFromModel(selector_model, prefit=True, threshold='median')
X_train_selected = selector.transform(X_train)

selected_feature_names = feature_names[selector.get_support()]
print(f"Selected {X_train_selected.shape[1]} features from an original of {X_train.shape[1]}.")

# --- 2. Hyperparameter Tuning ---
print("Performing hyperparameter tuning with GridSearchCV...")
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='balanced_accuracy',
    verbose=1
)
grid_search.fit(X_train_selected, y_train)

best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)


# --- ADD THIS SECTION to calculate training accuracy ---
print("\nCalculating accuracy on the training data...")
y_train_pred = best_model.predict(X_train_selected)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"   -> Accuracy on Training Data: {train_accuracy * 100:.2f}%")
# ----------------------------------------------------


# --- 3. Save the final model and artifacts ---
print("\nSaving model and artifacts...")
joblib.dump(best_model, MODEL_PATH)
joblib.dump(selector, SELECTOR_PATH)
joblib.dump(selected_feature_names, SELECTED_FEATURES_PATH)

print(f"✅ Model training complete.")
print(f"   -> Best model saved to: {MODEL_PATH}")
print(f"   -> Feature selector saved to: {SELECTOR_PATH}")
print(f"   -> Selected feature names saved to: {SELECTED_FEATURES_PATH}")
print("-" * 50 + "\n")