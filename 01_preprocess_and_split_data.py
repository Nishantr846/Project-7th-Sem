import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

print("--- 🚀 Stage 1: Preprocessing and Splitting Data ---")

# --- Configuration ---
RAW_DATA_PATH = 'data/symptom_disease_dataset.csv'
OUTPUT_DIR = 'outputs'
TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, 'train_data.joblib')
TEST_DATA_PATH = os.path.join(OUTPUT_DIR, 'test_data.joblib')
MLB_ENCODER_PATH = os.path.join(OUTPUT_DIR, 'mlb_encoder.joblib')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- NLTK Setup ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Helper Functions for Text Cleaning ---
def clean_symptom(symptom):
    """Normalizes and lemmatizes a single symptom string."""
    symptom = re.sub(r'[^a-zA-Z\s]', '', symptom.lower())
    words = symptom.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    cleaned_symptom = '_'.join(lemmatized_words)
    return cleaned_symptom

def preprocess_symptoms_text(text):
    """Extracts, cleans, and processes symptoms from a text string."""
    if not isinstance(text, str):
        return []
    
    # Simple split by comma for comma-separated lists
    symptoms = [s.strip() for s in text.split(',')]
    
    cleaned_symptoms = set()
    for symptom in symptoms:
        cleaned = clean_symptom(symptom)
        if cleaned:
            cleaned_symptoms.add(cleaned)
    return list(cleaned_symptoms)

# --- Main Logic ---
# Load Data
df = pd.read_csv(RAW_DATA_PATH)
df.columns = df.columns.str.strip()
df.dropna(subset=['text', 'label'], inplace=True)

# Preprocess Symptoms
print("Preprocessing symptom text...")
df['symptoms_list'] = df['text'].apply(preprocess_symptoms_text)

# Filter out rare diseases (less than 3 samples)
value_counts = df['label'].value_counts()
valid_labels = value_counts[value_counts >= 3].index
df_filtered = df[df['label'].isin(valid_labels)].reset_index(drop=True)

# Feature Engineering with MultiLabelBinarizer
print("Performing multi-hot encoding...")
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_filtered['symptoms_list'])
y = df_filtered['label']
feature_names = mlb.classes_

print(f"Created {X.shape[0]} samples with {X.shape[1]} unique features.")

# Split Data into Training and Testing Sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the processed data and the encoder
joblib.dump({'X_train': X_train, 'y_train': y_train, 'columns': feature_names}, TRAIN_DATA_PATH)
joblib.dump({'X_test': X_test, 'y_test': y_test, 'columns': feature_names}, TEST_DATA_PATH)
joblib.dump(mlb, MLB_ENCODER_PATH)

print(f"✅ Data successfully processed and saved.")
print(f"   -> Training data saved to: {TRAIN_DATA_PATH}")
print(f"   -> Testing data saved to: {TEST_DATA_PATH}")
print(f"   -> Encoder saved to: {MLB_ENCODER_PATH}")
print("-" * 50 + "\n")