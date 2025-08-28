#!/bin/bash
echo "Starting the ML Pipeline..."

# Run each script in sequence
python 01_preprocess_and_split_data.py && \
python 02_train_model.py && \
python 03_evaluate_model.py

echo "Pipeline finished."