from flask import Flask, request, jsonify, render_template
import joblib
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Initialize the Flask application
app = Flask(__name__)

# --- Download NLTK data if not present ---
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Define Paths and Load Artifacts ---
OUTPUT_DIR = 'outputs'

try:
    # Load the machine learning model
    model = joblib.load(os.path.join(OUTPUT_DIR, 'symptom_checker_model.joblib'))
    # Load the feature selector
    selector = joblib.load(os.path.join(OUTPUT_DIR, 'feature_selector.joblib'))
    # Load the list of selected symptom names for the dropdown
    selected_symptoms_list = joblib.load(os.path.join(OUTPUT_DIR, 'selected_feature_names.joblib'))
    # Load the fitted MultiLabelBinarizer
    mlb = joblib.load(os.path.join(OUTPUT_DIR, 'mlb_encoder.joblib'))
except FileNotFoundError as e:
    print(f"Error: A required file was not found. Details: {e}")
    print("Please run the full pipeline ('01_...', '02_...') to generate the necessary files in the 'outputs' directory.")
    exit()

# --- Preprocessing setup from the pipeline ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_symptom(symptom):
    """Normalizes and lemmatizes a single symptom string, matching the training pipeline."""
    symptom = re.sub(r'[^a-zA-Z\s]', '', symptom.lower())
    words = symptom.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # The pipeline uses underscores, so we replicate that here
    cleaned_symptom = '_'.join(lemmatized_words)
    return cleaned_symptom

# --- Disease to Doctor Specialty Mapping ---
specialty_map = {
    'Fungal infection': 'Dermatologist', 'Allergy': 'Allergist/Immunologist',
    'GERD': 'Gastroenterologist', 'Chronic cholestasis': 'Gastroenterologist',
    'Drug Reaction': 'Dermatologist', 'Peptic ulcer diseae': 'Gastroenterologist',
    'AIDS': 'Infectious Disease Specialist', 'Diabetes': 'Endocrinologist',
    'Gastroenteritis': 'Gastroenterologist', 'Bronchial Asthma': 'Pulmonologist',
    'Hypertension': 'Cardiologist', 'Migraine': 'Neurologist',
    'Cervical spondylosis': 'Orthopedist', 'Paralysis (brain hemorrhage)': 'Neurologist',
    'Jaundice': 'Gastroenterologist', 'Malaria': 'Infectious Disease Specialist',
    'Chicken pox': 'General Physician / Pediatrician', 'Dengue': 'Infectious Disease Specialist',
    'Typhoid': 'Infectious Disease Specialist', 'hepatitis A': 'Gastroenterologist',
    'Hepatitis B': 'Gastroenterologist', 'Hepatitis C': 'Gastroenterologist',
    'Hepatitis D': 'Gastroenterologist', 'Hepatitis E': 'Gastroenterologist',
    'Alcoholic hepatitis': 'Gastroenterologist', 'Tuberculosis': 'Pulmonologist',
    'Common Cold': 'General Physician', 'Pneumonia': 'Pulmonologist',
    'Dimorphic hemmorhoids(piles)': 'Proctologist', 'Heart attack': 'Cardiologist',
    'Varicose veins': 'Vascular Surgeon', 'Hypothyroidism': 'Endocrinologist',
    'Hyperthyroidism': 'Endocrinologist', 'Hypoglycemia': 'Endocrinologist',
    'Osteoarthristis': 'Orthopedist', 'Arthritis': 'Rheumatologist',
    '(vertigo) Paroymsal  Positional Vertigo': 'ENT Specialist', 'Acne': 'Dermatologist',
    'Urinary tract infection': 'Urologist', 'Psoriasis': 'Dermatologist',
    'Impetigo': 'Dermatologist', 'Default': 'General Physician'
}

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main page and populates the dropdown with selectable symptoms."""
    # Create a list of dictionaries for the template for better display formatting
    symptoms_for_template = [
        {'value': symptom, 'display': symptom.replace('_', ' ').title()}
        for symptom in selected_symptoms_list
    ]
    # Sort by the display name
    sorted_symptoms = sorted(symptoms_for_template, key=lambda x: x['display'])
    return render_template('index.html', symptoms=sorted_symptoms)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request, including full preprocessing."""
    if request.method == 'POST':
        # The list from the dropdown is already perfectly formatted
        symptoms_list = request.json.get('symptoms', [])
        
        if not symptoms_list:
            return jsonify({'error': 'No symptoms provided'}), 400

        # --- Replicate the Full Preprocessing Pipeline ---

        # 1. Use the symptom list directly (DO NOT re-clean it)
        #    This prevents the UserWarning.
        input_vector_full = mlb.transform([symptoms_list])
        
        # 2. Apply the feature selector
        input_vector_selected = selector.transform(input_vector_full)

        # --- Make Prediction ---
        prediction = model.predict(input_vector_selected)
        
        # 3. Decode the numeric prediction to the disease name
        #    This prevents the IndexError.
        predicted_label_index = prediction[0] 
        predicted_disease = model.classes_[predicted_label_index]

        # --- Get Recommended Specialty ---
        recommended_specialty = specialty_map.get(predicted_disease, 'General Physician')

        return jsonify({
            'predicted_disease': predicted_disease,
            'recommended_specialty': recommended_specialty
        })

if __name__ == '__main__':
    app.run(debug=True)