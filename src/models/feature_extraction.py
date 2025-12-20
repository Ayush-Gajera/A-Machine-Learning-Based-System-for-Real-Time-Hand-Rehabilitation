"""
Feature extraction utilities for hand exercise detection system.

This module contains functions for extracting features from MediaPipe
hand landmarks for machine learning classification.
"""

import numpy as np
import mediapipe as mp
import cv2


# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)


def calculate_angle(p1, p2, p3):
    """
    Calculate angle between three points.
    
    Args:
        p1, p2, p3: 2D points as numpy arrays
        
    Returns:
        float: Angle in degrees
    """
    a = p1 - p2
    b = p3 - p2
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    angle = np.arccos(dot_product / (norm_a * norm_b))
    return np.degrees(angle)


def extract_features(landmarks):
    """
    Extract features from hand landmarks.
    
    Extracts 229 features:
    - 190 distance features (Euclidean distances between all landmark pairs)
    - 39 angle features (angles between consecutive landmark triplets)
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        list: Feature vector with 229 values
    """
    features = []
    
    # Distance features (190 features)
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            p1 = np.array([landmarks[i].x, landmarks[i].y])
            p2 = np.array([landmarks[j].x, landmarks[j].y])
            distance = np.linalg.norm(p1 - p2)
            features.append(distance)
    
    # Angle features (39 features)
    for i in range(0, len(landmarks) - 2):
        p1 = np.array([landmarks[i].x, landmarks[i].y])
        p2 = np.array([landmarks[i + 1].x, landmarks[i + 1].y])
        p3 = np.array([landmarks[i + 2].x, landmarks[i + 2].y])
        angle = calculate_angle(p1, p2, p3)
        features.append(angle)
    
    return features


def predict_exercise(image, model, scaler):
    """
    Predict exercise from image using trained model.
    
    Args:
        image: Input image (BGR format)
        model: Trained classifier model
        scaler: Fitted StandardScaler
        
    Returns:
        tuple: (predictions, hand_landmarks_list)
            - predictions: List of predicted exercise names
            - hand_landmarks_list: List of detected hand landmarks
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    predictions = []
    hand_landmarks_list = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [landmark for landmark in hand_landmarks.landmark]
            features = extract_features(landmarks)
            features = scaler.transform([features])
            prediction = model.predict(features)[0]
            predictions.append(prediction)
            hand_landmarks_list.append(hand_landmarks)
    
    return predictions, hand_landmarks_list


def annotate_image(image, predictions, hand_landmarks_list):
    """
    Annotate image with predictions and hand landmarks.
    
    Args:
        image: Input image to annotate
        predictions: List of predicted exercise names
        hand_landmarks_list: List of detected hand landmarks
        
    Returns:
        numpy.ndarray: Annotated image
    """
    annotated_image = image.copy()
    
    if predictions:
        for i, prediction in enumerate(predictions):
            # Draw landmarks
            mp_drawing.draw_landmarks(
                annotated_image, 
                hand_landmarks_list[i], 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Display prediction
            start_x, start_y = 10, 30 + (i * 150)
            cv2.putText(
                annotated_image, 
                f"Prediction: {prediction}", 
                (start_x, start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )
    
    return annotated_image 