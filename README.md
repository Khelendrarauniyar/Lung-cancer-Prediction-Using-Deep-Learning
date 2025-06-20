# Lung Cancer Prediction

This project uses a neural network to predict the likelihood of lung cancer based on survey data. The model is trained using TensorFlow/Keras and can be exported in TensorFlow Lite format for deployment on edge devices.

## Features

- Data preprocessing (encoding, scaling, deduplication)
- Neural network model for binary classification
- Model evaluation (accuracy, classification report)
- Export to TensorFlow Lite

## Dataset

The dataset (`survey lung cancer.csv`) contains survey responses with features such as gender, age, smoking habits, and medical history, along with a label indicating lung cancer diagnosis.

## Usage

1. **Install dependencies:**
    ```sh
    pip install pandas scikit-learn tensorflow
    ```

2. **Train and export the model:**
    ```sh
    python lung_cancer_prediction.py
    ```

3. **Output:**
    - Model evaluation metrics are printed to the console.
    - The trained model is saved as `lung_cancer_model.tflite`.

## Files

- `lung_cancer_prediction.py`: Main script for data processing, model training, evaluation, and export.
- `survey lung cancer.csv`: Dataset used for training and evaluation.
- `lung_cancer_model.tflite`: Exported TensorFlow Lite model (generated after running the script).

## License

This project is for educational purposes.
