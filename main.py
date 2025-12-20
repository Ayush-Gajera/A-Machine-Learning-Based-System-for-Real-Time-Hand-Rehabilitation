"""
Hand Rehabilitation System - Main Application
"""

import os
import cv2
import numpy as np
import joblib
import time
from collections import deque
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import mediapipe as mp
from threading import Lock
from PIL import Image
from translations import TRANSLATIONS

# Global language setting
CURRENT_LANGUAGE = "en"

def get_text(key, lang=None):
    """Get translated text for the given key."""
    if lang is None:
        global CURRENT_LANGUAGE
        lang = CURRENT_LANGUAGE
    
    if key in TRANSLATIONS:
        return TRANSLATIONS[key].get(lang, TRANSLATIONS[key]["en"])
    return key


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
MODEL_FILE = 'exercise_classifier.pkl'
SCALER_FILE = 'scaler.pkl'

# Ensure directories exist
RESIZED_IMAGES_DIR = "static/images_resized"
os.makedirs(RESIZED_IMAGES_DIR, exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# Image paths mapping
EXERCISE_IMAGES = {
    "Ball-Grip-Wrist-Down": ["Ball_Grip_Wrist_Down.jpg", "Ball_Grip_Wrist_Down.png"],
    "Ball-Grip-Wrist-UP": ["Ball_Grip_Wrist_UP.jpg", "Ball_Grip_Wrist_Up.jpg", "Ball_Grip_Wrist_UP.png"],
    "Pinch": ["Pinch.jpg", "Pinch.png", "pinch.jpg", "pinch.png"],
    "Thumb-Extend": ["Thumb_Extend.jpg", "Thumb_Extend.png"],
    "Opposition": ["Opposition.jpg", "Opposition.png"],
    "Extend-Out": ["Extend_Out.jpg", "Extend_Out.png"],
    "Finger-Bend": ["Finger_Bend.jpg", "Finger_Bend.png"],
    "Side-Squzzer": ["Side_Squzzer.jpg", "Side_Squzzer.png", "Side_Squeezer.jpg"],
}

def resize_and_save_images():
    """Resize all exercise images to consistent size (350x250)."""
    source_dirs = ["images", "static/images"]
    target_size = (350, 250)
    
    for exercise, filenames in EXERCISE_IMAGES.items():
        found = False
        for source_dir in source_dirs:
            if found:
                break
            for filename in filenames:
                source_path = os.path.join(source_dir, filename)
                if os.path.exists(source_path):
                    try:
                        img = Image.open(source_path)
                        img = img.convert('RGB')
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        target_path = os.path.join(RESIZED_IMAGES_DIR, f"{exercise}.jpg")
                        img.save(target_path, 'JPEG', quality=90)
                        print(f"✓ Resized: {source_path} -> {target_path}")
                        found = True
                        break
                    except Exception as e:
                        print(f"✗ Error resizing {source_path}: {e}")
        if not found:
            print(f"✗ No image found for {exercise}")

# Resize images on startup
print("Resizing exercise images...")
resize_and_save_images()
print("Done!")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load ML model
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# Global finger tip variables
thumb_tip = None
index_finger_tip = None
middle_finger_tip = None
ring_finger_tip = None
pinky_finger_tip = None
thumb_ip = None

def update_finger_tips(landmarks):
    global thumb_tip, thumb_ip, index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_finger_tip
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_finger_tip = np.array([landmarks[8].x, landmarks[8].y])
    middle_finger_tip = np.array([landmarks[12].x, landmarks[12].y])
    ring_finger_tip = np.array([landmarks[16].x, landmarks[16].y])
    pinky_finger_tip = np.array([landmarks[20].x, landmarks[20].y])
    thumb_ip = np.array([landmarks[3].x, landmarks[3].y])

# ============= INTELLIGENT AUTO-TIMER =============
class IntelligentTimer:
    """
    Auto-detects when user holds good form and counts down.
    Resets automatically if exercise changes or form breaks.
    """
    def __init__(self, hold_duration=10):
        self.hold_duration = hold_duration
        self.enabled = False  # Toggle from UI
        self.current_exercise = None
        self.hold_start_time = None
        self.is_counting = False
        self.completed = False
        self.consecutive_good_frames = 0
        self.frames_to_start = 10  # Need ~0.3 sec good form to start
        
    def enable(self, enabled=True):
        self.enabled = enabled
        if not enabled:
            self._reset()
    
    def update(self, exercise, feedback_list):
        """Update timer based on exercise and feedback quality."""
        if not self.enabled:
            return self._get_status("disabled")
        
        # No hand detected
        if not exercise:
            self._reset()
            return self._get_status("no_hand")
        
        # Exercise changed - reset
        if exercise != self.current_exercise:
            self._reset()
            self.current_exercise = exercise
            return self._get_status("exercise_changed")
        
        # Check form quality
        is_good = self._is_good_form(feedback_list)
        
        if is_good:
            self.consecutive_good_frames += 1
            
            # Start counting after consistent good form
            if self.consecutive_good_frames >= self.frames_to_start and not self.is_counting and not self.completed:
                self.is_counting = True
                self.hold_start_time = time.time()
            
            # Check if completed
            if self.is_counting and not self.completed:
                elapsed = time.time() - self.hold_start_time
                if elapsed >= self.hold_duration:
                    self.completed = True
                    self.is_counting = False
                    return self._get_status("completed")
                return self._get_status("counting")
            
            if self.completed:
                return self._get_status("completed")
            
            return self._get_status("warming_up")
        else:
            # Bad form - reset counter but keep exercise
            self.consecutive_good_frames = 0
            if self.is_counting:
                self.is_counting = False
                self.hold_start_time = None
            return self._get_status("bad_form")
    
    def _is_good_form(self, feedback_list):
        """
        Check form quality based on original English feedback keys.
        """
        if not feedback_list:
            return False
        
        # Keywords to identify good/bad form in the original English keys
        good_words = ['good', 'great', 'properly', 'correctly', 'maintain', 'excellent', 'perfect', 'aligned', 'attached', 'maintaned']
        bad_words = ['try', 'keep', 'close', 'far', 'too', 'bend', 'squeeze', 'bring', 'adjust', 'move', 'apart']
        
        good = sum(1 for fb in feedback_list if any(w in fb.lower() for w in good_words))
        bad = sum(1 for fb in feedback_list if any(w in fb.lower() for w in bad_words))
        
        return good > bad
    
    def _reset(self):
        self.current_exercise = None
        self.hold_start_time = None
        self.is_counting = False
        self.completed = False
        self.consecutive_good_frames = 0
    
    def reset_for_next(self):
        """Reset for next exercise attempt (after completion)."""
        self.hold_start_time = None
        self.is_counting = False
        self.completed = False
        self.consecutive_good_frames = 0
    
    def _get_status(self, state):
        if state == "disabled":
            return {"enabled": False, "state": "disabled", "message": get_text("Enable Interactive Mode to use timer")}
        
        if state == "no_hand":
            return {"enabled": True, "state": "waiting", "message": get_text("Show your hand to camera"), "remaining": self.hold_duration, "progress": 0}
        
        if state == "exercise_changed":
            return {"enabled": True, "state": "waiting", "message": get_text("Exercise detected! Hold position..."), "remaining": self.hold_duration, "progress": 0}
        
        if state == "bad_form":
            return {"enabled": True, "state": "bad_form", "message": get_text("Improve form to start timer"), "remaining": self.hold_duration, "progress": 0}
        
        if state == "warming_up":
            progress = min(99, (self.consecutive_good_frames / self.frames_to_start) * 100)
            return {"enabled": True, "state": "warming_up", "message": get_text("Good! Hold steady..."), "remaining": self.hold_duration, "progress": progress}
        
        if state == "counting":
            elapsed = time.time() - self.hold_start_time
            remaining = max(0, self.hold_duration - elapsed)
            progress = min(100, (elapsed / self.hold_duration) * 100)
            msg = get_text("Hold! {}s remaining").format(int(remaining))
            return {"enabled": True, "state": "counting", "message": msg, "remaining": int(remaining), "progress": progress}
        
        if state == "completed":
            return {"enabled": True, "state": "completed", "message": get_text("🎉 Great job! Exercise complete!"), "remaining": 0, "progress": 100}
        
        return {"enabled": True, "state": "unknown", "message": get_text("..."), "remaining": self.hold_duration, "progress": 0}

# Global timer
intelligent_timer = IntelligentTimer(hold_duration=10)

# ============= PREDICTION STABILIZATION =============
class PredictionStabilizer:
    def __init__(self, window_size=20, confidence_threshold=0.7):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=window_size)
        self.last_stable_prediction = None
        self.stability_counter = 0
        self.min_stability_frames = 8
        
    def add_prediction(self, prediction):
        self.prediction_history.append(prediction)
        
        if len(self.prediction_history) < 5:
            return self.last_stable_prediction or prediction
        
        counts = {}
        for pred in self.prediction_history:
            counts[pred] = counts.get(pred, 0) + 1
        
        most_common = max(counts, key=counts.get)
        confidence = counts[most_common] / len(self.prediction_history)
        
        if confidence >= self.confidence_threshold:
            if most_common == self.last_stable_prediction:
                self.stability_counter += 1
            else:
                self.stability_counter = 1
            if self.stability_counter >= self.min_stability_frames:
                self.last_stable_prediction = most_common
        
        return self.last_stable_prediction if self.last_stable_prediction else most_common
    
    def reset(self):
        self.prediction_history.clear()
        self.last_stable_prediction = None
        self.stability_counter = 0

stabilizer = PredictionStabilizer()

# ============= FEATURE EXTRACTION =============
def calculate_angle(p1, p2, p3):
    a = p1 - p2
    b = p3 - p2
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a * norm_b == 0:
        return 0
    return np.degrees(np.arccos(np.clip(dot / (norm_a * norm_b), -1.0, 1.0)))

def extract_features(landmarks):
    features = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            p1 = np.array([landmarks[i].x, landmarks[i].y])
            p2 = np.array([landmarks[j].x, landmarks[j].y])
            features.append(np.linalg.norm(p1 - p2))
    for i in range(len(landmarks) - 2):
        p1 = np.array([landmarks[i].x, landmarks[i].y])
        p2 = np.array([landmarks[i + 1].x, landmarks[i + 1].y])
        p3 = np.array([landmarks[i + 2].x, landmarks[i + 2].y])
        features.append(calculate_angle(p1, p2, p3))
    return features

# ============= FEEDBACK FUNCTIONS =============
def provide_feedback_Ball_Grip_Wrist_Down(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    index_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
    
    d1 = np.linalg.norm(index_finger_tip - index_mcp)
    d2 = np.linalg.norm(middle_finger_tip - middle_mcp)
    
    if d1 < 0.055 and d2 < 0.055:
        feedback.append(get_text("Release the ball slowly."))
    elif d1 > 0.06 and d2 > 0.06:
        feedback.append(get_text("Squeeze the ball tightly."))
    else:
        feedback.append(get_text("Good grip maintained."))

    if np.linalg.norm(thumb_tip - index_finger_tip) < 0.05 and np.linalg.norm(thumb_tip - middle_finger_tip) < 0.05:
        feedback.append(get_text("Good thumb position for a strong grip."))
    else:
        feedback.append(get_text("Adjust your thumb position for a better grip."))

    d_im = np.linalg.norm(index_finger_tip - middle_finger_tip)
    d_mr = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    d_rp = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    if d_im < 0.02: feedback.append(get_text("Index and middle fingers are too close."))
    elif d_im > 0.05: feedback.append(get_text("Index and middle fingers are too far apart."))
    else: feedback.append(get_text("Index and middle fingers are correctly positioned."))

    if d_mr < 0.02: feedback.append(get_text("Middle and ring fingers are too close."))
    elif d_mr > 0.05: feedback.append(get_text("Middle and ring fingers are too far apart."))
    else: feedback.append(get_text("Middle and ring fingers are correctly positioned."))

    if d_rp < 0.02: feedback.append(get_text("Ring and pinky fingers are too close."))
    elif d_rp > 0.05: feedback.append(get_text("Ring and pinky fingers are too far apart."))
    else: feedback.append(get_text("Ring and pinky fingers are correctly positioned."))

    return feedback

def provide_feedback_Ball_Grip_Wrist_UP(landmarks):
    return provide_feedback_Ball_Grip_Wrist_Down(landmarks)  # Same logic

def provide_feedback_Pinch(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    
    pinch_dist = np.linalg.norm(thumb_tip - index_finger_tip)
    if pinch_dist > 0.17:
        feedback.append(get_text("Try to bring your thumb and index finger closer."))
    else:
        feedback.append(get_text("Good pinch! Maintain the grip."))

    d_im = np.linalg.norm(index_finger_tip - middle_finger_tip)
    d_mr = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    d_rp = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    if d_im < 0.01: feedback.append(get_text("Index and middle fingers are too close."))
    elif d_im > 0.05: feedback.append(get_text("Index and middle fingers are too far apart."))
    else: feedback.append(get_text("Index and middle fingers are correctly positioned."))

    if d_mr < 0.01: feedback.append(get_text("Middle and ring fingers are too close."))
    elif d_mr > 0.05: feedback.append(get_text("Middle and ring fingers are too far apart."))
    else: feedback.append(get_text("Middle and ring fingers are correctly positioned."))

    if d_rp < 0.01: feedback.append(get_text("Ring and pinky fingers are too close."))
    elif d_rp > 0.07: feedback.append(get_text("Ring and pinky fingers are too far apart."))
    else: feedback.append(get_text("Ring and pinky fingers are correctly positioned."))

    return feedback

def provide_feedback_Thumb_Extend(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    index_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
    ring_mcp = np.array([landmarks[13].x, landmarks[13].y])

    if np.linalg.norm(thumb_ip - index_mcp) > 0.07:
        feedback.append(get_text("Thumb center is far from index finger base; squeeze tighter."))
    else:
        feedback.append(get_text("Good distance between thumb center and index finger base."))

    if np.linalg.norm(thumb_ip - middle_mcp) >= 0.065:
        feedback.append(get_text("Thumb center is far from middle finger base; move closer."))
    else:
        feedback.append(get_text("Good thumb center position relative to middle finger base."))

    if np.linalg.norm(thumb_tip - index_mcp) > 0.085:
        feedback.append(get_text("Thumb tip is too far from index finger base; bring closer."))
    else:
        feedback.append(get_text("Good thumb tip position relative to index finger base."))

    if np.linalg.norm(thumb_tip - middle_mcp) > 0.08:
        feedback.append(get_text("Thumb tip is too far from middle finger base; bring closer."))
    else:
        feedback.append(get_text("Good thumb tip position relative to middle finger base."))

    return feedback

def provide_feedback_Opposition(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    index_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
    ring_mcp = np.array([landmarks[13].x, landmarks[13].y])
    
    if np.linalg.norm(thumb_ip - index_mcp) > 0.095:
        feedback.append(get_text("Thumb center is far from index finger base; squeeze tighter."))
    else:
        feedback.append(get_text("Good distance between thumb center and index finger base."))
    
    if np.linalg.norm(thumb_ip - middle_mcp) >= 0.06:
        feedback.append(get_text("Thumb center is far from middle finger base; move closer."))
    else:
        feedback.append(get_text("Good thumb center position relative to middle finger base."))
    
    if np.linalg.norm(thumb_tip - index_mcp) > 0.1:
        feedback.append(get_text("Thumb tip is too far from index finger base; bring closer."))
    else:
        feedback.append(get_text("Good thumb tip position relative to index finger base."))
    
    if np.linalg.norm(thumb_tip - middle_mcp) > 0.09:
        feedback.append(get_text("Thumb tip is too far from middle finger base; bring closer."))
    else:
        feedback.append(get_text("Good thumb tip position relative to middle finger base."))
    
    return feedback

def provide_feedback_Extend_Out(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    index_mcp = np.array([landmarks[5].x, landmarks[5].y])
    ring_dip = np.array([landmarks[15].x, landmarks[15].y])
    
    d_im = np.linalg.norm(index_finger_tip - middle_finger_tip)
    d_mr = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    d_ti = np.linalg.norm(thumb_tip - index_mcp)
    d_pr = np.linalg.norm(ring_dip - pinky_finger_tip)
    
    if d_im >= 0.05: feedback.append(get_text("Keep index and middle finger attached!"))
    else: feedback.append(get_text("Index and middle finger are properly attached."))
    
    if d_mr >= 0.07: feedback.append(get_text("Keep middle and ring finger attached!"))
    else: feedback.append(get_text("Middle and ring finger are properly attached."))
    
    if d_ti <= 0.06: feedback.append(get_text("Keep thumb and index finger base far apart!"))
    elif d_ti <= 0.15: feedback.append(get_text("Good distance maintained for thumb."))
    else: feedback.append(get_text("Thumb is too far; bend it and keep close!"))
    
    if d_pr <= 0.08: feedback.append(get_text("Keep ring finger and pinky far apart!"))
    elif d_pr <= 0.14: feedback.append(get_text("Good distance maintained for pinky finger."))
    else: feedback.append(get_text("Pinky is too far from ring finger; keep close!"))
    
    return feedback

def provide_feedback_Finger_Bend(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    
    d_im = np.linalg.norm(index_finger_tip - middle_finger_tip)
    d_mr = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    d_rp = np.linalg.norm(ring_finger_tip - pinky_finger_tip)
    d_ti = np.linalg.norm(thumb_tip - index_finger_tip)
    d_tm = np.linalg.norm(thumb_tip - middle_finger_tip)
    d_tr = np.linalg.norm(thumb_tip - ring_finger_tip)
    d_tp = np.linalg.norm(thumb_tip - pinky_finger_tip)
    
    if d_im >= 0.06: feedback.append(get_text("Keep index and middle finger close!"))
    else: feedback.append(get_text("Index and middle finger are properly aligned."))
    
    if d_mr >= 0.06: feedback.append(get_text("Keep middle and ring finger close!"))
    else: feedback.append(get_text("Middle and ring finger are properly aligned."))
    
    if d_rp >= 0.06: feedback.append(get_text("Keep ring and pinky finger close!"))
    else: feedback.append(get_text("Ring and pinky finger are properly aligned."))
    
    if d_ti >= 0.085: feedback.append(get_text("Keep index finger and thumb close!"))
    else: feedback.append(get_text("Index finger and thumb are properly aligned."))
    
    if d_tm >= 0.085: feedback.append(get_text("Keep middle finger and thumb close!"))
    else: feedback.append(get_text("Middle finger and thumb are properly aligned."))
    
    if d_tr >= 0.085: feedback.append(get_text("Keep ring finger and thumb close!"))
    else: feedback.append(get_text("Ring finger and thumb are properly aligned."))
    
    if d_tp >= 0.085: feedback.append(get_text("Keep pinky finger and thumb close!"))
    else: feedback.append(get_text("Pinky finger and thumb are properly aligned."))
    
    return feedback

def provide_feedback_Side_Squzzer(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    
    d_im = np.linalg.norm(index_finger_tip - middle_finger_tip)
    if d_im > 0.05:
        feedback.append(get_text("Squeeze tighter between index and middle finger."))
    else:
        feedback.append(get_text("Great squeeze! Now release and repeat."))
    
    t_tip = np.array([landmarks[4].x, landmarks[4].y])
    i_pip = np.array([landmarks[6].x, landmarks[6].y])
    d_thumb = min(np.linalg.norm(t_tip - i_pip), np.linalg.norm(t_tip - middle_finger_tip))
    
    if d_thumb >= 0.045:
        feedback.append(get_text("Keep thumb attached with squeezing fingers."))
    else:
        feedback.append(get_text("Good thumb position with squeezing fingers."))
    
    ring_mcp = np.array([landmarks[13].x, landmarks[13].y])
    pinky_mcp = np.array([landmarks[17].x, landmarks[17].y])
    
    if np.linalg.norm(ring_finger_tip - ring_mcp) >= 0.04:
        feedback.append(get_text("Bend your ring finger more inward."))
    else:
        feedback.append(get_text("Good bending of ring finger."))
    
    if np.linalg.norm(pinky_finger_tip - pinky_mcp) >= 0.04:
        feedback.append(get_text("Bend your pinky finger more inward."))
    else:
        feedback.append(get_text("Good bending of pinky finger."))
    
    return feedback

def default_feedback(landmarks):
    return [get_text("Position your hand clearly in front of the camera.")]

FEEDBACK_FUNCTIONS = {
    "Ball-Grip-Wrist-Down": provide_feedback_Ball_Grip_Wrist_Down,
    "Ball-Grip-Wrist-UP": provide_feedback_Ball_Grip_Wrist_UP,
    "Pinch": provide_feedback_Pinch,
    "Thumb-Extend": provide_feedback_Thumb_Extend,
    "Opposition": provide_feedback_Opposition,
    "Extend-Out": provide_feedback_Extend_Out,
    "Finger-Bend": provide_feedback_Finger_Bend,
    "Side-Squzzer": provide_feedback_Side_Squzzer,
}

# ============= VIDEO CAMERA =============
class VideoCamera:
    def __init__(self):
        self.video = None
        self.is_running = False
        self.current_exercise = None
        self.current_feedback = []
        self.lock = Lock()
        self.timer_status = {}
        
    def start(self):
        if not self.is_running:
            self.video = cv2.VideoCapture(0)
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            stabilizer.reset()
            
    def stop(self):
        if self.video:
            self.video.release()
        self.is_running = False
        self.current_exercise = None
        self.current_feedback = []
        self.timer_status = {}
        intelligent_timer._reset()
        
    def get_frame(self):
        if not self.is_running or not self.video:
            return None
            
        success, frame = self.video.read()
        if not success:
            return None
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        exercise = None
        feedback_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                landmarks = list(hand_landmarks.landmark)
                features = extract_features(landmarks)
                features_scaled = scaler.transform([features])
                
                raw_pred = model.predict(features_scaled)[0]
                exercise = stabilizer.add_prediction(raw_pred)
                
                if exercise:
                    feedback_func = FEEDBACK_FUNCTIONS.get(exercise, default_feedback)
                    # Get raw English feedback for logic processing
                    raw_feedback = feedback_func(landmarks)
                    
                    # Update timer using RAW English feedback
                    timer_status = intelligent_timer.update(exercise, raw_feedback)
                    
                    # Translate feedback for the UI/TTS just before saving
                    # Return list of dicts: {'text': translated, 'type': 'good'|'warning'}
                    translated_feedback = []
                    for fb in raw_feedback:
                        is_good = any(w in fb.lower() for w in ['good', 'great', 'properly', 'correctly', 'maintain', 'excellent', 'perfect', 'aligned', 'attached', 'maintaned'])
                        translated_feedback.append({
                            'text': get_text(fb),
                            'type': 'good' if is_good else 'warning'
                        })
                    
                    with self.lock:
                        self.current_exercise = exercise
                        self.current_feedback = translated_feedback
                        self.timer_status = timer_status
        else:
            timer_status = intelligent_timer.update(None, [])
            with self.lock:
                self.timer_status = timer_status
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

camera = VideoCamera()

# ============= FLASK ROUTES =============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while camera.is_running:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/images_resized/<path:filename>')
def serve_resized_image(filename):
    return send_from_directory(RESIZED_IMAGES_DIR, filename)

@app.route('/start_camera', methods=['POST'])
def start_camera():
    camera.start()
    return jsonify({'status': 'started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    camera.stop()
    return jsonify({'status': 'stopped'})

@app.route('/toggle_interactive', methods=['POST'])
def toggle_interactive():
    data = request.json
    enabled = data.get('enabled', False)
    intelligent_timer.enable(enabled)
    return jsonify({'status': 'success', 'enabled': enabled})

@app.route('/reset_timer', methods=['POST'])
def reset_timer():
    intelligent_timer.reset_for_next()
    return jsonify({'status': 'reset'})

@app.route('/set_language', methods=['POST'])
def set_language():
    data = request.json
    lang = data.get('language', 'en')
    global CURRENT_LANGUAGE
    CURRENT_LANGUAGE = lang
    return jsonify({'status': 'success', 'language': CURRENT_LANGUAGE})

@app.route('/get_status')
def get_status():
    with camera.lock:
        return jsonify({
            'exercise': camera.current_exercise,
            'feedback': camera.current_feedback,
            'is_running': camera.is_running,
            'timer': camera.timer_status
        })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("=" * 60)
    print("   Hand Rehabilitation System")
    print("   Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
