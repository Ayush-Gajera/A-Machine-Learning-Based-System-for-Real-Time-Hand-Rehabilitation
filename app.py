import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
CSV_FILE = 'landmarks_features.csv'
MODEL_FILE = 'exercise_classifier.pkl'
SCALER_FILE = 'scaler.pkl'
CONFUSION_MATRIX_FILE = 'confusion_matrix.png'

# Load or train model
def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        st.success('Loaded existing model.')
    else:
        data = pd.read_csv(CSV_FILE, header=None)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0].apply(lambda x: os.path.basename(x).split('_')[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = RandomForestClassifier(n_estimators=100, random_state=22)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Model Accuracy: {accuracy * 100:.2f}%')
        plot_confusion_matrix(y_test, y_pred, model.classes_)
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        st.success('Trained and saved new model.')
    return model, scaler

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close()
    st.image(CONFUSION_MATRIX_FILE)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

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
        p2 = np.array([landmarks[i + 1].x, landmarks[i + 1].y])
        p3 = np.array([landmarks[i + 2].x, landmarks[i + 2].y])
        angle = calculate_angle(p1, p2, p3)
        features.append(angle)
    return features

def calculate_angle(p1, p2, p3):
    a = p1 - p2
    b = p3 - p2
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    angle = np.arccos(dot_product / (norm_a * norm_b))
    return np.degrees(angle)

def default_feedback(landmarks):
    return ["Feedback is not available for this exercise."]

thumb_tip = None
index_finger_tip = None
middle_finger_tip = None
ring_finger_tip = None
pinky_finger_tip = None
thumb_ip=None

def update_finger_tips(landmarks):
    global thumb_tip,thumb_ip, index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_finger_tip
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_finger_tip = np.array([landmarks[8].x, landmarks[8].y])
    middle_finger_tip = np.array([landmarks[12].x, landmarks[12].y])
    ring_finger_tip = np.array([landmarks[16].x, landmarks[16].y])
    pinky_finger_tip = np.array([landmarks[20].x, landmarks[20].y])
    thumb_ip = np.array([landmarks[3].x, landmarks[3].y])

# Feedback Function for "Ball_Grip_Wrist_Down"
def provide_feedback_Ball_Grip_Wrist_Down(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])
    distance_index_tip_to_mcp = np.linalg.norm(index_finger_tip - index_finger_mcp)
    distance_middle_tip_to_mcp = np.linalg.norm(middle_finger_tip - middle_finger_mcp)
    if distance_index_tip_to_mcp < 0.055 and distance_middle_tip_to_mcp < 0.055:  # Threshold for a tight grip
        feedback.append("Release the ball slowly.")
    elif distance_index_tip_to_mcp > 0.06 and distance_middle_tip_to_mcp > 0.06:  # Threshold for a loose grip
        feedback.append("Squeeze the ball tightly.")
    else:
        feedback.append("Maintain your grip.")

    # distance from thumb to index and middle fingers
    distance_thumb_to_index_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_thumb_to_middle_tip = np.linalg.norm(thumb_tip - middle_finger_tip)

    # feedback on thumb position relative to index and middle fingers
    if distance_thumb_to_index_tip < 0.05 and distance_thumb_to_middle_tip < 0.05:
        feedback.append("Good thumb position for a strong grip.")
    else:
        feedback.append("Adjust your thumb position for a better grip.")

    # distances between neighboring fingertips
    distance_index_to_middle_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_middle_to_ring_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_ring_to_pinky_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # feedback on the position of index and middle fingers
    if distance_index_to_middle_tip < 0.02:
        feedback.append("Index and middle fingers are too close.")
    elif distance_index_to_middle_tip > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    # feedback on the position of middle and ring fingers
    if distance_middle_to_ring_tip < 0.02:
        feedback.append("Middle and ring fingers are too close.")
    elif distance_middle_to_ring_tip > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    # feedback on the position of ring and pinky fingers
    if distance_ring_to_pinky_tip < 0.02:
        feedback.append("Ring and pinky fingers are too close.")
    elif distance_ring_to_pinky_tip > 0.05:
        feedback.append("Ring and pinky fingers are too far apart.")
    else:
        feedback.append("Ring and pinky fingers are correctly positioned.")

    return feedback

# Feedback Function for "Ball_Grip_Wrist_UP"
def provide_feedback_Ball_Grip_Wrist_UP(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    # coordinates for MCP joints of index and middle fingers
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])

    # distance from the MCP joint to the fingertip for both index and middle fingers
    distance_index_tip_to_mcp = np.linalg.norm(index_finger_tip - index_finger_mcp)
    distance_middle_tip_to_mcp = np.linalg.norm(middle_finger_tip - middle_finger_mcp)

    # feedback based on the distance between MCP and fingertip for index and middle fingers
    if distance_index_tip_to_mcp < 0.055 and distance_middle_tip_to_mcp < 0.055:  # Threshold for a tight grip
        feedback.append("Release the ball slowly.")
    elif distance_index_tip_to_mcp > 0.06 and distance_middle_tip_to_mcp > 0.06:  # Threshold for a loose grip
        feedback.append("Squeeze the ball tightly.")
    else:
        feedback.append("Maintain your grip.")

    # distance from thumb to index and middle fingers
    distance_thumb_to_index_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_thumb_to_middle_tip = np.linalg.norm(thumb_tip - middle_finger_tip)

    # feedback on thumb position relative to index and middle fingers
    if distance_thumb_to_index_tip < 0.05 and distance_thumb_to_middle_tip < 0.05:
        feedback.append("Good thumb position for a strong grip.")
    else:
        feedback.append("Adjust your thumb position for a better grip.")

    # distances between neighboring fingertips
    distance_index_to_middle_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_middle_to_ring_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_ring_to_pinky_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # feedback on the position of index and middle fingers
    if distance_index_to_middle_tip < 0.02:
        feedback.append("Index and middle fingers are too close.")
    elif distance_index_to_middle_tip > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    # feedback on the position of middle and ring fingers
    if distance_middle_to_ring_tip < 0.02:
        feedback.append("Middle and ring fingers are too close.")
    elif distance_middle_to_ring_tip > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    # feedback on the position of ring and pinky fingers
    if distance_ring_to_pinky_tip < 0.02:
        feedback.append("Ring and pinky fingers are too close.")
    elif distance_ring_to_pinky_tip > 0.05:
        feedback.append("Ring and pinky fingers are too far apart.")
    else:
        feedback.append("Ring and pinky fingers are correctly positioned.")

    return feedback

# Feedback Function for "Pinch"
def provide_feedback_Pinch(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    # distance between thumb and index finger tips
    pinch_distance = np.linalg.norm(thumb_tip - index_finger_tip)
    # print(pinch_distance)  # Debug print to check the pinch distance

    # Determine the state of the pinch based on the distance
    if pinch_distance > 0.17:  # Threshold for a loose pinch
        feedback.append("Try to bring your thumb and index finger closer.")
    else:
        feedback.append("Good pinch! Maintain the grip.")

    # Feedback on finger positions relative to their neighbors
    index_to_middle_distance = np.linalg.norm(index_finger_tip - middle_finger_tip)
    middle_to_ring_distance = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    ring_to_pinky_distance = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # Example thresholds for finger position feedback
    if index_to_middle_distance < 0.01:
        feedback.append("Index and middle fingers are too close.")
    elif index_to_middle_distance > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    if middle_to_ring_distance < 0.01:
        feedback.append("Middle and ring fingers are too close.")
    elif middle_to_ring_distance > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    if ring_to_pinky_distance < 0.01:
        feedback.append("Ring and pinky fingers are too close.")
    elif ring_to_pinky_distance > 0.07:
        feedback.append("Ring and pinky fingers are too far apart.")
    else:
        feedback.append("Ring and pinky fingers are correctly positioned.")

    return feedback

# Feedback Function for "Thumb Extend"
def provide_feedback_Thumb_Extend(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    # Get the coordinates for the thumb IP, thumb tip, and MCPs of index, middle, and ring fingers
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])
    ring_finger_mcp = np.array([landmarks[13].x, landmarks[13].y])

    # distances from the thumb tip to the index, middle, and ring MCPs
    thumb_tip_to_index_mcp_distance = np.linalg.norm(thumb_tip - index_finger_mcp)
    thumb_tip_to_middle_mcp_distance = np.linalg.norm(thumb_tip - middle_finger_mcp)
    thumb_tip_to_ring_mcp_distance = np.linalg.norm(thumb_tip - ring_finger_mcp)
    
    # distances from the thumb IP to the index, middle, and ring MCPs
    thumb_ip_to_index_mcp_distance = np.linalg.norm(thumb_ip - index_finger_mcp)
    thumb_ip_to_middle_mcp_distance = np.linalg.norm(thumb_ip - middle_finger_mcp)
    thumb_ip_to_ring_mcp_distance = np.linalg.norm(thumb_ip - ring_finger_mcp)
    
    # Feedback for thumb IP to index MCP
    if thumb_ip_to_index_mcp_distance > 0.07:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from index finger base; try to keep it closer by squeezing tighter.")
    else:
        feedback.append("Good distance maintained between thumb center and base of index finger.")

    # Feedback for thumb IP to middle MCP
    if thumb_ip_to_middle_mcp_distance >= 0.065:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from the middle finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the middle finger base.")

    # Feedback for thumb IP to ring MCP
    if thumb_ip_to_ring_mcp_distance >= 0.095:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from the ring finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the ring finger base.")

    # Feedback for thumb tip to index MCP
    if thumb_tip_to_index_mcp_distance > 0.085:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from index finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the index finger base.")

    # Feedback for thumb tip to middle MCP
    if thumb_tip_to_middle_mcp_distance > 0.08:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from middle finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the middle finger base.")

    # Feedback for thumb tip to ring MCP
    if thumb_tip_to_ring_mcp_distance > 0.064:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from ring finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the ring finger base.")
    return feedback

# Feedback Function for "Opposition"
def provide_feedback_Opposition(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])
    ring_finger_mcp = np.array([landmarks[13].x, landmarks[13].y])
    thumb_tip_to_index_mcp_distance = np.linalg.norm(thumb_tip - index_finger_mcp)
    thumb_tip_to_middle_mcp_distance = np.linalg.norm(thumb_tip - middle_finger_mcp)
    thumb_tip_to_ring_mcp_distance = np.linalg.norm(thumb_tip - ring_finger_mcp)
    thumb_ip_to_index_mcp_distance = np.linalg.norm(thumb_ip - index_finger_mcp)
    thumb_ip_to_middle_mcp_distance = np.linalg.norm(thumb_ip - middle_finger_mcp)
    thumb_ip_to_ring_mcp_distance = np.linalg.norm(thumb_ip - ring_finger_mcp)
    # print(thumb_tip_to_index_mcp_distance)  print(thumb_tip_to_middle_mcp_distance)  print(thumb_tip_to_ring_mcp_distance)  print(thumb_ip_to_index_mcp_distance)  print(thumb_ip_to_middle_mcp_distance)  print(thumb_ip_to_ring_mcp_distance)  print("----")
    if thumb_ip_to_index_mcp_distance > 0.095:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from index finger base; try to keep it closer by squeezing tighter.")
    else:
        feedback.append("Good distance maintained between thumb center and base of index finger.")
    if thumb_ip_to_middle_mcp_distance >= 0.06:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from the middle finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the middle finger base.")
    if thumb_ip_to_ring_mcp_distance >= 0.045:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from the ring finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the ring finger base.")
    if thumb_tip_to_index_mcp_distance > 0.1:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from index finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the index finger base.")
    if thumb_tip_to_middle_mcp_distance > 0.09:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from middle finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the middle finger base.")
    if thumb_tip_to_ring_mcp_distance > 0.11:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from ring finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the ring finger base.")
    return feedback

# Feedback Function for "Extend Out"
def provide_feedback_Extend_Out(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    ring_finger_dip = np.array([landmarks[15].x, landmarks[15].y])
    update_finger_tips(landmarks)
    distance_between_index_tip_and_middle_finger_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_between_middle_tip_and_ring_finger_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_between_thumb_tip_and_index_finger_mcp = np.linalg.norm(thumb_tip - index_finger_mcp)
    distance_between_pinky_finger_tip_and_rinf_finger_dip = np.linalg.norm(ring_finger_dip - pinky_finger_tip)
    # print(distance_between_index_tip_and_middle_finger_tip)  print(distance_between_middle_tip_and_ring_finger_tip)  print(distance_between_thumb_tip_and_index_finger_mcp)  print(distance_between_pinky_finger_tip_and_rinf_finger_dip)
    if distance_between_index_tip_and_middle_finger_tip >= 0.05:
        feedback.append("Keep index finger and middle finger attached with each other!")
    else:
        feedback.append("Index finger and middle finger are properly attached.")
    if distance_between_middle_tip_and_ring_finger_tip >= 0.07:
        feedback.append("Keep middle finger and ring finger attached with each other!")
    else:
        feedback.append("middle finger and ring finger are properly attached.")
    if distance_between_thumb_tip_and_index_finger_mcp <= 0.06:
        feedback.append("Keep thumb and index finger base far from each other!")
    elif distance_between_thumb_tip_and_index_finger_mcp >= 0.061 and distance_between_thumb_tip_and_index_finger_mcp <= 0.15:
        feedback.append("Good distance maintainance for thumb.")
    else:
        feedback.append("Thumb is very far from index finger base so bend it and keep close!")
    if distance_between_pinky_finger_tip_and_rinf_finger_dip <= 0.08:
        feedback.append("Keep ring finger upper joint and pinky finger far from each other!")
    elif distance_between_pinky_finger_tip_and_rinf_finger_dip > 0.081 and distance_between_pinky_finger_tip_and_rinf_finger_dip <= 0.14:
        feedback.append("Good distance maintainance for pinky finger.")
    else:
        feedback.append("Pinky finger is very far from ring finger keep it close!")
    return feedback

# Feedback Function for "Finger Bend"
def provide_feedback_Finger_Bend(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    distance_between_index_tip_and_middle_finger_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_between_middle_tip_and_ring_finger_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_between_ring_tip_and_pinky_finger_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)
    distance_between_thumb_tip_and_index_finger_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_between_thumb_tip_and_middle_finger_tip = np.linalg.norm(thumb_tip - middle_finger_tip)
    distance_between_thumb_tip_and_ring_finger_tip = np.linalg.norm(thumb_tip - ring_finger_tip)
    distance_between_thumb_tip_and_pinky_finger_tip = np.linalg.norm(thumb_tip - pinky_finger_tip)
    # print(distance_between_index_tip_and_middle_finger_tip)  print(distance_between_middle_tip_and_ring_finger_tip) print(distance_between_ring_tip_and_pinky_finger_tip) print(distance_between_thumb_tip_and_index_finger_tip) print(distance_between_thumb_tip_and_middle_finger_tip)  print(distance_between_thumb_tip_and_ring_finger_tip) print(distance_between_thumb_tip_and_pinky_finger_tip)
    if distance_between_index_tip_and_middle_finger_tip >= 0.06:
        feedback.append("Keep index finger and middle finger close to each other!")
    else:
        feedback.append("index finger and middle finger are properly align.")
    if distance_between_middle_tip_and_ring_finger_tip >= 0.06:
        feedback.append("Keep middle finger and ring finger close to each other!")
    else:
        feedback.append("middle finger and ring finger are properly align.")
    if distance_between_ring_tip_and_pinky_finger_tip >= 0.06:
        feedback.append("Keep ring finger and pinky finger close to each other!")
    else:
        feedback.append("ring finger and pinky finger are properly align.")
    if distance_between_thumb_tip_and_index_finger_tip >= 0.085:
        feedback.append("Keep index finger and thumb close to each other!")
    else:
        feedback.append("index finger and thumb are properly align.")
    if distance_between_thumb_tip_and_middle_finger_tip >= 0.085:
        feedback.append("Keep middle finger and thumb close to each other!")
    else:
        feedback.append("middle finger and thumb are properly align.")
    if distance_between_thumb_tip_and_ring_finger_tip >= 0.085:
        feedback.append("Keep ring finger and thumb close to each other!")
    else:
        feedback.append("ring finger and thumb are properly align.")
    if distance_between_thumb_tip_and_pinky_finger_tip >= 0.085:
        feedback.append("Keep pinky finger and thumb close to each other!")
    else:
        feedback.append("pinky finger and thumb are properly align.")
    return feedback

# Feedback Function for "Side Squeezer"
def provide_feedback_Side_Squzzer(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    distance_between_tips = np.linalg.norm(index_finger_tip - middle_finger_tip)
    if distance_between_tips > 0.05:  # Example threshold value for close proximity
        feedback.append("Try to squeeze the ball more tightly and make the distance min. between index and middle finger.")
    else:
        feedback.append("Great job maintaining a tight squeeze now release and repeat it.")   
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_finger_pip=np.array([landmarks[6].x, landmarks[6].y])
    thumb_distance_to_squeezing_fingers = min(
        np.linalg.norm(thumb_tip - index_finger_pip),
        np.linalg.norm(thumb_tip - middle_finger_tip)
    )
    # print(thumb_distance_to_squeezing_fingers)
    if thumb_distance_to_squeezing_fingers >= 0.045:  # Example threshold for thumb interference
        feedback.append("Keep your thumb attched with squeezing fingers.")
    else:
        feedback.append("Good thumb position with squeezing fingers. Keep them attached.")
    ring_finger_mcp = np.array([landmarks[13].x, landmarks[13].y])
    pinky_finger_mcp = np.array([landmarks[17].x, landmarks[17].y])
    ring_finger_distance_to_ring_finger_mcp = np.linalg.norm(ring_finger_tip - ring_finger_mcp)
    # print(ring_finger_distance_to_ring_finger_mcp)
    if ring_finger_distance_to_ring_finger_mcp >= 0.04:
        feedback.append("Try to bend your ring finger more inward.")
    else:
        feedback.append("Good bending of ring finger.")
    pinky_finger_distance_to_pinky_finger_mcp = np.linalg.norm(pinky_finger_tip - pinky_finger_mcp)
    # print(pinky_finger_distance_to_pinky_finger_mcp)    
    if pinky_finger_distance_to_pinky_finger_mcp >= 0.04:
        feedback.append("Try to bend your pinky finger more inward.")
    else:
        feedback.append("Good bending of pinky finger.")        
    return feedback

# Function to predict exercise and feedback
def predict_exercise(image, model, scaler):
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

# Annotate the image with predictions and feedback
def annotate_image(image, predictions, hand_landmarks_list):
    annotated_image = image.copy()
    if predictions:
        for i, prediction in enumerate(predictions):
            feedback_function_name = f'provide_feedback_{prediction.replace("-", "_")}'
            feedback_function = globals().get(feedback_function_name, default_feedback)
            feedback = feedback_function(hand_landmarks_list[i].landmark)
            
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks_list[i], mp_hands.HAND_CONNECTIONS)
            
            # Display predictions and feedback on the image
            start_x, start_y = 10, 30 + (i * 150)  # Adjust the vertical offset for each hand detected
            cv2.putText(annotated_image, f"Prediction: {prediction}", (start_x, start_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Limit to a few feedback messages to avoid crowding the image
            for j, fb in enumerate(feedback[:5]):  # Display up to 5 feedback messages
                cv2.putText(annotated_image, f"- {fb}", (start_x, start_y + 25 * (j + 1)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated_image


# Streamlit UI
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import os

# Paths
DEMO_VIDEO_PATH = "demo-video.mp4"  # Replace with the actual path to your video file
EXERCISE_IMAGES = {
    "Ball_Grip_Wrist_Down": "images/Ball_Grip_Wrist_Down.jpg",
    "Ball_Grip_Wrist_Up": "images/Ball_Grip_Wrist_UP.jpg",
    "Pinch": "images/pinch.png",
    "Thumb_Extend": "images/Thumb_Extend.jpg",
    "Opposition": "images/Opposition.jpg",
    "Extend_Out": "images/Extend_Out.png",
    "Finger_Bend": "images/Finger_Bend.png",
    "Side_Squzzer": "images/Side_Squzzer.png"
}

# Function to autoplay video in Streamlit
def autoplay_video(video_path):
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
        base64_video = base64.b64encode(video_bytes).decode("utf-8")
        st.markdown(
            f"""
            <video autoplay loop width="700" height="350">
                <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
            </video>
            """,
            unsafe_allow_html=True,
        )

# Streamlit UI
import cv2
import streamlit as st
from PIL import Image

# Function to resize images
def resize_image(image_path, width, height):
    try:
        img = Image.open(image_path)
        img = img.resize((width, height))  # Resize to consistent dimensions
        return img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Streamlit UI
def main():
    st.set_page_config(layout="wide", page_title=" Hand Rehabilitation System")
    st.title("Hand Rehabilitation System")
    
    # CSS for smooth transition and consistent image size
    st.markdown("""
        <style>
        .main-view-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
        }
        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #f7f9fc;
            padding: 10px;
            border-radius: 10px;
        }
        .video-container video {
            width: 480px;  /* Adjusted width for compact size */
            height: 200px; /* Adjusted height maintaining 16:9 ratio */
            border-radius: 10px;
        }
        .reference-image {
            transition: opacity 0.5s ease-in-out; /* Smooth transition */
            max-width: 600px;
            max-height: 400px;
            display: block;
            margin: auto;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Main Application Sections
    st.header("Demonstration Video")
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_path = "demo-video.mp4"  # Replace with your video file path
    st.video(video_path)
    st.markdown('</div>', unsafe_allow_html=True)

    st.header("Live Hand Detection")
    
    # Create layout containers
    col1, col2 = st.columns([2, 1])
    with col1:
        FRAME_WINDOW = st.image([])
    with col2:
        st.subheader("Reference Image")
        default_image = resize_image("all_exe.png", width=450, height=350)  
        exercise_image = st.image(default_image, caption="All exercises")  
    
    # Load the model and scaler
    model, scaler = load_or_train_model()

    # Checkbox to start detection
    run = st.checkbox('Run Hand Detection')

    if run:
        cap = cv2.VideoCapture(0)  # Open the webcam
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break
            
            predictions, hand_landmarks_list = predict_exercise(frame, model, scaler)
            annotated_frame = annotate_image(frame, predictions, hand_landmarks_list)

            # Display the live feed
            FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Update the reference image based on the detected exercise
            if predictions:
                exercise_name = predictions[0].replace("-", "_")
                reference_image_path = f"images/{exercise_name}.jpg"
                resized_image = resize_image(reference_image_path, width=450, height=350)  # Resize image
                if resized_image:
                    exercise_image.image(resized_image, caption=f"Reference: {predictions[0]}", use_column_width=False)
                else:
                    exercise_image.image(default_image, caption="No reference available", use_column_width=False)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
