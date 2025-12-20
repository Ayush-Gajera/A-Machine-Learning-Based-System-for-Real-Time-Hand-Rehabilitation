"""
Model utilities for hand exercise detection system.

This module contains functions for loading, training, and evaluating
the machine learning model used for exercise classification.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Import configuration
from ..utils.config import MODEL_FILE, SCALER_FILE, CSV_FILE, CONFUSION_MATRIX_FILE


def load_or_train_model():
    """
    Load existing model or train a new one if not available.
    
    Returns:
        tuple: (model, scaler) - Trained model and fitted scaler
    """
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        print('Loaded existing model.')
    else:
        # Check if CSV file exists
        if not os.path.exists(CSV_FILE):
            print(f'Training data not found at {CSV_FILE}')
            print('Please ensure the training data is available.')
            return None, None
            
        # Load and prepare data
        data = pd.read_csv(CSV_FILE, header=None)
        X = data.iloc[:, 1:]  # Features (all columns except first)
        y = data.iloc[:, 0].apply(lambda x: os.path.basename(x).split('_')[0])  # Labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=22)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')
        
        # Generate confusion matrix
        plot_confusion_matrix(y_test, y_pred, model.classes_)
        
        # Save model and scaler
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        print('Trained and saved new model.')
    
    return model, scaler


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot and save confusion matrix for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close()
    print(f'Confusion matrix saved as {CONFUSION_MATRIX_FILE}')


def export_model_to_tflite(model, output_path='exercise_classifier.tflite'):
    """
    Export scikit-learn model to TensorFlow Lite format.
    
    Args:
        model: Trained scikit-learn model
        output_path: Output path for TFLite model
    """
    try:
        import tensorflow as tf
        
        # Create a dummy TensorFlow model that wraps the scikit-learn model
        class SklearnWrapper(tf.keras.layers.Layer):
            def __init__(self, sklearn_model):
                super(SklearnWrapper, self).__init__()
                self.sklearn_model = sklearn_model

            def call(self, inputs):
                if tf.is_tensor(inputs):
                    inputs = tf.convert_to_tensor(inputs)
                predictions = self.sklearn_model.predict(inputs)
                return tf.convert_to_tensor(predictions)

        # Wrap and convert the model
        sklearn_wrapper = SklearnWrapper(model)
        concrete_func = tf.function(sklearn_wrapper.call).get_concrete_function(
            tf.TensorSpec(shape=[None, 229], dtype=tf.float32)
        )

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()

        # Save the TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f'Model exported to TFLite format: {output_path}')
        
    except ImportError:
        print("TensorFlow not available. Skipping TFLite export.")
    except Exception as e:
        print(f"Error exporting to TFLite: {e}") 