import joblib
from sklearn.metrics import accuracy_score, classification_report
import os

print("--- 🚀 Stage 3: Evaluating Model ---")

# --- Configuration ---
OUTPUT_DIR = 'outputs'
TEST_DATA_PATH = os.path.join(OUTPUT_DIR, 'test_data.joblib')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'symptom_checker_model.joblib')
SELECTOR_PATH = os.path.join(OUTPUT_DIR, 'feature_selector.joblib')

# --- 1. Load test data and the trained model ---
print("Loading test data, model, and feature selector...")
test_data = joblib.load(TEST_DATA_PATH)
X_test = test_data['X_test']
y_test = test_data['y_test']

model = joblib.load(MODEL_PATH)
selector = joblib.load(SELECTOR_PATH)

# --- 2. Apply the same feature selection to the test data ---
X_test_selected = selector.transform(X_test)
print(f"Test data transformed to {X_test_selected.shape[1]} features.")

# --- 3. Make predictions and evaluate ---
print("Making predictions on the test set...")
y_pred = model.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n" + "="*50)
print("          Model Performance Evaluation")
print("="*50)
print(f"\nAccuracy on Test Set: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(report)
print("="*50)

print("✅ Evaluation complete.")
print("-" * 50 + "\n")