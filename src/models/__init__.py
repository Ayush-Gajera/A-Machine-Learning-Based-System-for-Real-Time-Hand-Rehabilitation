"""
Models package for hand exercise detection system.
Contains machine learning model utilities and training functions.
"""

from .model_utils import load_or_train_model, plot_confusion_matrix
from .feature_extraction import extract_features, calculate_angle, predict_exercise, annotate_image

__all__ = [
    'load_or_train_model',
    'plot_confusion_matrix',
    'extract_features',
    'calculate_angle',
    'predict_exercise',
    'annotate_image'
] 