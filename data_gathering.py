import os
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for
import mediapipe as mp

app = Flask(__name__)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Directories to save images
IMAGE_DIR = 'static/images'
ANNOTATED_DIR = 'static/annotated-images'
CSV_FILE = 'landmarks_features2.csv'

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Initialize CSV file for landmarks if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w') as f:
        f.write('image_path,features\n')  # Adding a header to the CSV file for clarity

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Global state for recording
is_recording = False
current_exercise = ''

def calculate_angle(p1, p2, p3):
    a = p1 - p2
    b = p3 - p2
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    angle = np.arccos(dot_product / (norm_a * norm_b))
    return np.degrees(angle)

def extract_features(landmarks):
    features = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            p1 = np.array([landmarks[i].x, landmarks[i].y])
            p2 = np.array([landmarks[j].x, landmarks[j].y])
            distance = np.linalg.norm(p1 - p2)
            features.append(distance)

    for i in range(0, len(landmarks) - 2):
        p1 = np.array([landmarks[i].x, landmarks[i].y])
        p2 = np.array([landmarks[i+1].x, landmarks[i+1].y])
        p3 = np.array([landmarks[i+2].x, landmarks[i+2].y])
        angle = calculate_angle(p1, p2, p3)
        features.append(angle)

    return features

def save_features_to_csv(image_path, features):
    features_str = ','.join(map(str, features))
    try:
        with open(CSV_FILE, 'a') as f:
            f.write(f"{image_path},{features_str}\n")
    except PermissionError as e:
        print(f"PermissionError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def annotate_and_save(image, image_path):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [landmark for landmark in hand_landmarks.landmark]
            features = extract_features(landmarks)
            save_features_to_csv(image_path, features)

    if current_exercise:
        annotated_image_path = os.path.join(ANNOTATED_DIR, f'{current_exercise}_{os.path.basename(image_path)}')
        cv2.imwrite(annotated_image_path, image)

def gen_frames():
    global is_recording
    while True:
        success, frame = cap.read()
        if not success:
            break
        if is_recording:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            image_path = os.path.join(IMAGE_DIR, f'{current_exercise}_{timestamp}.jpg')
            cv2.imwrite(image_path, frame)
            annotate_and_save(frame, image_path)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, current_exercise
    current_exercise = request.form.get('exercise')
    is_recording = True
    return redirect(url_for('index'))

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
