"""
Configuration constants for hand exercise detection system.
"""

# Video and image paths
DEMO_VIDEO_PATH = "demo-video.mp4"

# Exercise reference images mapping
EXERCISE_IMAGES = {
    "Ball_Grip_Wrist_Down": "images/Ball_Grip_Wrist_Down.jpg",
    "Ball_Grip_Wrist_UP": "images/Ball_Grip_Wrist_UP.jpg",
    "Pinch": "images/Pinch.jpg",
    "Thumb_Extend": "images/Thumb_Extend.jpg",
    "Opposition": "images/Opposition.jpg",
    "Extend_Out": "images/Extend_Out.jpg",
    "Finger_Bend": "images/Finger_Bend.jpg",
    "Side_Squzzer": "images/Side_Squzzer.jpg"
}

# Supported exercises
SUPPORTED_EXERCISES = [
    "Ball_Grip_Wrist_Down",
    "Ball_Grip_Wrist_UP", 
    "Pinch",
    "Thumb_Extend",
    "Opposition",
    "Extend_Out",
    "Finger_Bend",
    "Side_Squzzer"
]

# Model configuration
MODEL_CONFIG = {
    'n_estimators': 100,
    'random_state': 22,
    'test_size': 0.2,
    'train_random_state': 40
}

# MediaPipe configuration
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 2,
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.7
}

# File paths
MODEL_FILE = 'models/exercise_classifier.pkl'
SCALER_FILE = 'models/scaler.pkl'
CSV_FILE = 'data/landmarks_features.csv'
CONFUSION_MATRIX_FILE = 'confusion_matrix.png' 