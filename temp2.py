import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import mediapipe as mp
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r".*SymbolDatabase.GetPrototype\(\) is deprecated.*")
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
        print('Loaded existing model.')
    else:
        data = pd.read_csv(CSV_FILE, header=None)
        X = data.iloc[:, 1:]
        # Extract exercise name from the image path
        y = data.iloc[:, 0].apply(lambda x: os.path.basename(x).split('_')[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = RandomForestClassifier(n_estimators=100, random_state=22)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')
        plot_confusion_matrix(y_test, y_pred, model.classes_)
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        print('Trained and saved new model.')
    return model, scaler
# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close()
    print(f'Confusion matrix saved as {CONFUSION_MATRIX_FILE}')
# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
def calculate_angle(p1, p2, p3):
    a = p1 - p2
    b = p3 - p2
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    angle = np.arccos(dot_product / (norm_a * norm_b))
    return np.degrees(angle)
# Function to extract features from hand landmarks (228 features)
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
# Default feedback function if not available
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

    # Calculate the distance from thumb to index and middle fingers
    distance_thumb_to_index_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_thumb_to_middle_tip = np.linalg.norm(thumb_tip - middle_finger_tip)

    # Provide feedback on thumb position relative to index and middle fingers
    if distance_thumb_to_index_tip < 0.05 and distance_thumb_to_middle_tip < 0.05:
        feedback.append("Good thumb position for a strong grip.")
    else:
        feedback.append("Adjust your thumb position for a better grip.")

    # Calculate the distances between neighboring fingertips
    distance_index_to_middle_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_middle_to_ring_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_ring_to_pinky_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # Provide feedback on the position of index and middle fingers
    if distance_index_to_middle_tip < 0.02:
        feedback.append("Index and middle fingers are too close.")
    elif distance_index_to_middle_tip > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    # Provide feedback on the position of middle and ring fingers
    if distance_middle_to_ring_tip < 0.02:
        feedback.append("Middle and ring fingers are too close.")
    elif distance_middle_to_ring_tip > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    # Provide feedback on the position of ring and pinky fingers
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
    # Calculate the coordinates for MCP joints of index and middle fingers
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])

    # Calculate the distance from the MCP joint to the fingertip for both index and middle fingers
    distance_index_tip_to_mcp = np.linalg.norm(index_finger_tip - index_finger_mcp)
    distance_middle_tip_to_mcp = np.linalg.norm(middle_finger_tip - middle_finger_mcp)

    # Provide feedback based on the distance between MCP and fingertip for index and middle fingers
    if distance_index_tip_to_mcp < 0.055 and distance_middle_tip_to_mcp < 0.055:  # Threshold for a tight grip
        feedback.append("Release the ball slowly.")
    elif distance_index_tip_to_mcp > 0.06 and distance_middle_tip_to_mcp > 0.06:  # Threshold for a loose grip
        feedback.append("Squeeze the ball tightly.")
    else:
        feedback.append("Maintain your grip.")

    # Calculate the distance from thumb to index and middle fingers
    distance_thumb_to_index_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_thumb_to_middle_tip = np.linalg.norm(thumb_tip - middle_finger_tip)

    # Provide feedback on thumb position relative to index and middle fingers
    if distance_thumb_to_index_tip < 0.05 and distance_thumb_to_middle_tip < 0.05:
        feedback.append("Good thumb position for a strong grip.")
    else:
        feedback.append("Adjust your thumb position for a better grip.")

    # Calculate the distances between neighboring fingertips
    distance_index_to_middle_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_middle_to_ring_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_ring_to_pinky_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # Provide feedback on the position of index and middle fingers
    if distance_index_to_middle_tip < 0.02:
        feedback.append("Index and middle fingers are too close.")
    elif distance_index_to_middle_tip > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    # Provide feedback on the position of middle and ring fingers
    if distance_middle_to_ring_tip < 0.02:
        feedback.append("Middle and ring fingers are too close.")
    elif distance_middle_to_ring_tip > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    # Provide feedback on the position of ring and pinky fingers
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
    # Calculate the distance between thumb and index finger tips
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

    # Calculate the distances from the thumb tip to the index, middle, and ring MCPs
    thumb_tip_to_index_mcp_distance = np.linalg.norm(thumb_tip - index_finger_mcp)
    thumb_tip_to_middle_mcp_distance = np.linalg.norm(thumb_tip - middle_finger_mcp)
    thumb_tip_to_ring_mcp_distance = np.linalg.norm(thumb_tip - ring_finger_mcp)
    
    # Calculate the distances from the thumb IP to the index, middle, and ring MCPs
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
def predict_exercise(image, model, scaler):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    predictions = []
    hand_landmarks_list = []
    confidences = []  # Store confidences
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [landmark for landmark in hand_landmarks.landmark]
            features = extract_features(landmarks)  # 229 features
            features = scaler.transform([features])  # Scale features
            probabilities = model.predict_proba(features)[0]  # Get prediction probabilities
            prediction = model.predict(features)[0]  # Predict exercise
            confidence = max(probabilities)  # Get the highest probability as confidence

            # Adjust confidence if it fluctuates around 95%
            # if confidence < 0.95 and confidence>0.85:
                # confidence = 0.75  # Lower the displayed confidence
            confidences.append(confidence)  # Store confidence
            predictions.append(prediction)
            hand_landmarks_list.append(hand_landmarks)
    return predictions, hand_landmarks_list, confidences

def annotate_image(image, predictions, hand_landmarks_list, confidences):
    if predictions:
        for i, prediction in enumerate(predictions):
            # Display the exercise prediction
            cv2.putText(image, prediction, (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the confidence level
            confidence_text = f"Confidence: {confidences[i] * 100:.2f}%"
            cv2.putText(image, confidence_text, (10, 60 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            # Get the feedback function
            formatted_prediction = prediction.replace("-", "_")
            feedback_function_name = f'provide_feedback_{formatted_prediction}'
            if feedback_function_name in globals():
                feedback_function = globals()[feedback_function_name]
            else:
                feedback_function = default_feedback

            # Provide feedback
            feedback = feedback_function(hand_landmarks_list[i].landmark)  # Only pass landmarks
            font_scale = 0.6
            line_spacing = 15
            for j, fb in enumerate(feedback):
                y_position = 100 + i * (len(feedback) * line_spacing) + j * line_spacing
                cv2.putText(image, fb, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks_list[i], mp_hands.HAND_CONNECTIONS)
    return image

def main():
    model, scaler = load_or_train_model()
    cap = cv2.VideoCapture(1)  # Open the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            continue
        predictions, hand_landmarks_list, confidences = predict_exercise(frame, model, scaler)
        annotated_frame = annotate_image(frame, predictions, hand_landmarks_list, confidences)
        cv2.imshow('Hand Exercise Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
