# import os
# import cv2
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import mediapipe as mp
# import joblib
# import time

# # Paths
# CSV_FILE = 'landmarks_features.csv'
# MODEL_FILE = 'exercise_classifier.pkl'
# SCALER_FILE = 'scaler.pkl'
# CONFUSION_MATRIX_FILE = 'confusion_matrix.png'

# # Mapping of joint indices to names
# joint_names = {
#     0: 'Wrist',
#     1: 'Thumb_CMC',
#     2: 'Thumb_MCP',
#     3: 'Thumb_IP',
#     4: 'Thumb_Tip',
#     5: 'Index_Finger_MCP',
#     6: 'Index_Finger_PIP',
#     7: 'Index_Finger_DIP',
#     8: 'Index_Finger_Tip',
#     9: 'Middle_Finger_MCP',
#     10: 'Middle_Finger_PIP',
#     11: 'Middle_Finger_DIP',
#     12: 'Middle_Finger_Tip',
#     13: 'Ring_Finger_MCP',
#     14: 'Ring_Finger_PIP',
#     15: 'Ring_Finger_DIP',
#     16: 'Ring_Finger_Tip',
#     17: 'Pinky_MCP',
#     18: 'Pinky_PIP',
#     19: 'Pinky_DIP',
#     20: 'Pinky_Tip'
# }

# # Load or train model
# def load_or_train_model():
#     if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
#         model = joblib.load(MODEL_FILE)
#         scaler = joblib.load(SCALER_FILE)
#         print('Loaded existing model.')
#     else:
#         # Load the data without column names
#         data = pd.read_csv(CSV_FILE, header=None)

#         # Split the image path (first column) and features (remaining columns)
#         X = data.iloc[:, 1:]
#         y = data.iloc[:, 0].apply(lambda x: x.split('_')[0])  # Assuming the exercise name is the prefix of the filename

#         # Split into training and testing datasets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Standardize the features
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)

#         # Initialize and train the model
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)

#         # Evaluate the model
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         print(f'Model Accuracy: {accuracy * 100:.2f}%')

#         # Plot confusion matrix and save it
#         plot_confusion_matrix(y_test, y_pred, model.classes_)

#         # Save the model and scaler
#         joblib.dump(model, MODEL_FILE)
#         joblib.dump(scaler, SCALER_FILE)
#         print('Trained and saved new model.')
    
#     return model, scaler

# # Function to plot confusion matrix
# def plot_confusion_matrix(y_true, y_pred, class_names):
#     cm = confusion_matrix(y_true, y_pred, labels=class_names)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.savefig(CONFUSION_MATRIX_FILE)
#     plt.close()
#     print(f'Confusion matrix saved as {CONFUSION_MATRIX_FILE}')

# # Initialize MediaPipe Hands model
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# # Function to calculate angle between three points
# def calculate_angle(p1, p2, p3):
#     a = p1 - p2
#     b = p3 - p2
#     dot_product = np.dot(a, b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     angle = np.arccos(dot_product / (norm_a * norm_b))
#     return np.degrees(angle)

# # Function to extract features from hand landmarks
# def extract_features(landmarks):
#     features = []
#     for i in range(len(landmarks)):
#         for j in range(i + 1, len(landmarks)):
#             p1 = np.array([landmarks[i].x, landmarks[i].y])
#             p2 = np.array([landmarks[j].x, landmarks[j].y])
#             distance = np.linalg.norm(p1 - p2)
#             features.append(distance)
#     for i in range(0, len(landmarks) - 2):
#         p1 = np.array([landmarks[i].x, landmarks[i].y])
#         p2 = np.array([landmarks[i + 1].x, landmarks[i + 1].y])
#         p3 = np.array([landmarks[i + 2].x, landmarks[i + 2].y])
#         angle = calculate_angle(p1, p2, p3)
#         features.append(angle)
#     return features

# # Dynamically determine relevant landmarks for the given exercise
# # Dynamically determine relevant landmarks for the given exercise
# def get_relevant_landmarks(exercise_name):
#     # Read CSV file
#     exercise_data = pd.read_csv(CSV_FILE, header=None)
#     exercise_data.columns = ['Exercise'] + [f'L{i}' for i in range(1, len(exercise_data.columns))]
    
#     # Filter data for the specific exercise name
#     specific_data = exercise_data[exercise_data['Exercise'].str.contains(exercise_name, regex=False)]
#     # print(f"Exercise Data for {exercise_name}:\n", specific_data)  # Debug: Check filtered data

#     # Calculate variance of each feature and extract top landmarks
#     variance = specific_data.iloc[:, 1:].var().sort_values(ascending=False)
#     # print(f"Variance of landmarks for {exercise_name}:\n", variance)  # Debug: Check variance calculation
    
#     # Extract top landmarks within the valid range of 0 to 20
#     top_landmarks = variance.head(22).index.str.extract(r'(\d+)').astype(int).to_numpy().flatten()
    
#     # Map large indices to a valid range (0-20) using modulo operation
#     top_landmarks = [idx % 21 for idx in top_landmarks]  # Modulo operation to keep indices in range
#     # print(f"Mapped Top landmarks for {exercise_name}:\n", top_landmarks)  # Debug: Check top landmarks after mapping
    
#     return top_landmarks

# # Provide feedback based on predicted exercise and landmarks
# # Provide feedback based on predicted exercise and landmarks
# def provide_feedback(predicted_exercise, landmarks):
#     # Get relevant landmarks based on the exercise name
#     relevant_indices = get_relevant_landmarks(predicted_exercise)
    
#     if len(relevant_indices) < 5:
#         print("Insufficient landmarks to provide feedback.")  # Debug: Check for insufficient landmarks
#         return "Insufficient landmarks for providing feedback."

#     # Calculate feedback using the relevant landmarks
#     feedback = []
#     for idx in relevant_indices:
#         landmark = landmarks[idx]
#         # Perform checks based on landmarks and provide feedback
#         # Placeholder feedback logic
#         if landmark.y < 0.5:
#             feedback.append(f"Landmark {idx} is too low.")
#         else:
#             feedback.append(f"Landmark {idx} is in the correct range.")
    
#     return feedback



# # Predict exercise from new image
# def predict_exercise(image, model, scaler):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)
#     predictions = []
#     hand_landmarks_list = []

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             landmarks = [landmark for landmark in hand_landmarks.landmark]
#             features = extract_features(landmarks)
#             features = scaler.transform([features])
#             prediction = model.predict(features)[0]
#             predictions.append(prediction)
#             hand_landmarks_list.append(hand_landmarks)
    
#     return predictions, hand_landmarks_list

# # Annotate image with prediction and feedback
# # Annotate image with prediction and feedback
# def annotate_image(image, predictions, hand_landmarks_list):
#     if predictions:
#         for i, prediction in enumerate(predictions):
#             cv2.putText(image, prediction, (10, 30 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             feedback = provide_feedback(prediction, hand_landmarks_list[i].landmark)
#             for j, fb in enumerate(feedback):
#                 cv2.putText(image, fb, (10, 70 + i*100 + j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#             mp_drawing.draw_landmarks(image, hand_landmarks_list[i], mp_hands.HAND_CONNECTIONS)
#     return image


# # Live feed capture and prediction
# def main():
#     model, scaler = load_or_train_model()
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame from webcam.")
#             break

#         predictions, hand_landmarks_list = predict_exercise(frame, model, scaler)
#         frame = annotate_image(frame, predictions, hand_landmarks_list)
        
#         cv2.imshow('Hand Exercise Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

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
import time

# Paths
CSV_FILE = 'landmarks_features.csv'
MODEL_FILE = 'exercise_classifier.pkl'
SCALER_FILE = 'scaler.pkl'
CONFUSION_MATRIX_FILE = 'confusion_matrix.png'

# Mapping of joint indices to names
joint_names = {
    0: 'Wrist',
    1: 'Thumb_CMC',
    2: 'Thumb_MCP',
    3: 'Thumb_IP',
    4: 'Thumb_Tip',
    5: 'Index_Finger_MCP',
    6: 'Index_Finger_PIP',
    7: 'Index_Finger_DIP',
    8: 'Index_Finger_Tip',
    9: 'Middle_Finger_MCP',
    10: 'Middle_Finger_PIP',
    11: 'Middle_Finger_DIP',
    12: 'Middle_Finger_Tip',
    13: 'Ring_Finger_MCP',
    14: 'Ring_Finger_PIP',
    15: 'Ring_Finger_DIP',
    16: 'Ring_Finger_Tip',
    17: 'Pinky_MCP',
    18: 'Pinky_PIP',
    19: 'Pinky_DIP',
    20: 'Pinky_Tip'
}

# Load or train model
def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        print('Loaded existing model.')
    else:
        # Load the data without column names
        data = pd.read_csv(CSV_FILE, header=None)

        # Split the image path (first column) and features (remaining columns)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0].apply(lambda x: x.split('_')[0])  # Assuming the exercise name is the prefix of the filename

        # Split into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')

        # Plot confusion matrix and save it
        plot_confusion_matrix(y_test, y_pred, model.classes_)

        # Save the model and scaler
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

# Function to calculate angle between three points
def calculate_angle(p1, p2, p3):
    a = p1 - p2
    b = p3 - p2
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    angle = np.arccos(dot_product / (norm_a * norm_b))
    return np.degrees(angle)

# Function to extract features from hand landmarks
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

# Dynamically determine relevant landmarks for the given exercise
# Dynamically determine relevant landmarks for the given exercise
def get_relevant_landmarks(exercise_name):
    # Read CSV file
    exercise_data = pd.read_csv(CSV_FILE, header=None)
    exercise_data.columns = ['Exercise'] + [f'L{i}' for i in range(1, len(exercise_data.columns))]
    
    # Filter data for the specific exercise name
    specific_data = exercise_data[exercise_data['Exercise'].str.contains(exercise_name, regex=False)]
    # print(f"Exercise Data for {exercise_name}:\n", specific_data)  # Debug: Check filtered data

    # Calculate variance of each feature and extract top landmarks
    variance = specific_data.iloc[:, 1:].var().sort_values(ascending=False)
    # print(f"Variance of landmarks for {exercise_name}:\n", variance)  # Debug: Check variance calculation
    
    # Extract top landmarks within the valid range of 0 to 20
    top_landmarks = variance.head(24).index.str.extract(r'(\d+)').astype(int).to_numpy().flatten()
    
    # Map large indices to a valid range (0-20) using modulo operation
    top_landmarks = [idx % 21 for idx in top_landmarks]  # Modulo operation to keep indices in range
    # print(f"Mapped Top landmarks for {exercise_name}:\n", top_landmarks)  # Debug: Check top landmarks after mapping
    
    return top_landmarks

# Provide feedback based on predicted exercise and landmarks
# Provide feedback based on predicted exercise and landmarks
def provide_feedback(predicted_exercise, landmarks):
    # Get relevant landmarks based on the exercise name
    relevant_indices = get_relevant_landmarks(predicted_exercise)
    
    if len(relevant_indices) < 5:
        print("Insufficient landmarks to provide feedback.")  # Debug: Check for insufficient landmarks
        return "Insufficient landmarks for providing feedback."

    # Calculate feedback using the relevant landmarks
    feedback = []
    for idx in relevant_indices:
        landmark = landmarks[idx]
        # Perform checks based on landmarks and provide feedback
        # Placeholder feedback logic
        if landmark.y < 0.5:
            feedback.append(f"Landmark {idx} is too low.")
        else:
            feedback.append(f"Landmark {idx} is in the correct range.")
    
    return feedback



# Predict exercise from new image
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

# Annotate image with prediction and feedback
# Annotate image with prediction and feedback
def annotate_image(image, predictions, hand_landmarks_list):
    if predictions:
        for i, prediction in enumerate(predictions):
            # Display the predicted exercise name at the top
            cv2.putText(image, prediction, (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            feedback = provide_feedback(prediction, hand_landmarks_list[i].landmark)
            
            # Set font size and line spacing
            font_scale = 0.4  # Reduced font scale for better fit
            line_spacing = 15  # Line spacing for feedback text
            
            # Loop through feedback and display it
            for j, fb in enumerate(feedback):
                y_position = 60 + i * (len(feedback) * line_spacing) + j * line_spacing  # Adjusted y position
                cv2.putText(image, fb, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks_list[i], mp_hands.HAND_CONNECTIONS)
    
    return image

# Live feed capture and prediction
def main():
    model, scaler = load_or_train_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        predictions, hand_landmarks_list = predict_exercise(frame, model, scaler)
        frame = annotate_image(frame, predictions, hand_landmarks_list)
        
        cv2.imshow('Hand Exercise Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
