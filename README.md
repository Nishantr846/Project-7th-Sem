# Symptom-Based Disease Prediction System

This project implements a machine learning-based system for predicting diseases based on symptoms. It includes data preprocessing, model training, evaluation, and a web interface for making predictions.

## Project Structure

```
├── data/
│   └── symptom_disease_dataset.csv    # Input dataset
├── outputs/                           # Trained models and artifacts
│   ├── feature_selector.joblib
│   ├── mlb_encoder.joblib
│   ├── selected_feature_names.joblib
│   ├── symptom_checker_model.joblib
│   ├── test_data.joblib
│   └── train_data.joblib
├── templates/
│   └── index.html                     # Web interface template
├── 01_preprocess_and_split_data.py    # Data preparation script
├── 02_train_model.py                  # Model training script
├── 03_evaluate_model.py               # Model evaluation script
├── app.py                             # Flask web application
├── requirements.txt                    # Project dependencies
└── run_pipeline.sh                    # Pipeline execution script
```

## Setup Instructions

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Option 1: Running the Complete Pipeline

To run the entire machine learning pipeline (preprocessing, training, and evaluation):

```bash
chmod +x run_pipeline.sh  # Make the script executable
./run_pipeline.sh
```

### Option 2: Running Individual Components

1. Preprocess and split the data:
   ```bash
   python 01_preprocess_and_split_data.py
   ```

2. Train the model:
   ```bash
   python 02_train_model.py
   ```

3. Evaluate the model:
   ```bash
   python 03_evaluate_model.py
   ```

### Running the Web Application

To start the web interface for making predictions:

```bash
python app.py
```

Then open your web browser and navigate to `http://localhost:5000` to use the symptom checker interface.

## Data

The project uses a symptom-disease dataset located in `data/symptom_disease_dataset.csv`. The preprocessed data and trained models are saved in the `outputs/` directory.

## Model Artifacts

The model files are not included in the repository due to their large size. After running the training pipeline, the following model artifacts will be generated in the `outputs/` directory:
- `feature_selector.joblib`: Feature selection model
- `mlb_encoder.joblib`: Multi-label encoder for symptoms/diseases
- `selected_feature_names.joblib`: Selected feature names
- `symptom_checker_model.joblib`: Trained disease prediction model
- `test_data.joblib`: Test dataset
- `train_data.joblib`: Training dataset

To generate these files, run the complete pipeline as described in the "Running the Project" section above.

## Web Interface

The web interface (`templates/index.html`) provides a user-friendly way to input symptoms and receive disease predictions based on the trained model.
