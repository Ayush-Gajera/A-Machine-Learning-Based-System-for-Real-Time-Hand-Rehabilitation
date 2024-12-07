import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to calculate angle between three points
def calculate_angle(p1, p2, p3):
    a = p1 - p2
    b = p3 - p2
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    angle = np.arccos(dot_product / (norm_a * norm_b))
    return np.degrees(angle)

# Function to provide feedback based on "Side Squeezer" exercise
def provide_feedback_side_squeezer(landmarks):
    feedback = []

    # Calculate the angle between Middle Finger MCP and Index Finger MCP
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])
    wrist = np.array([landmarks[0].x, landmarks[0].y])

    angle_between_fingers = calculate_angle(wrist, index_finger_mcp, middle_finger_mcp)
    
    # Feedback on angle between fingers
    if angle_between_fingers > 30:  # Example threshold value
        feedback.append("Try to bring your index and middle fingers closer together.")
    else:
        feedback.append("Good squeeze between index and middle fingers.")

    # Calculate the distance between Index Finger Tip and Middle Finger Tip
    index_finger_tip = np.array([landmarks[8].x, landmarks[8].y])
    middle_finger_tip = np.array([landmarks[12].x, landmarks[12].y])
    distance_between_tips = np.linalg.norm(index_finger_tip - middle_finger_tip)

    # Feedback on distance between finger tips
    if distance_between_tips > 0.05:  # Example threshold value for close proximity
        feedback.append("Try to squeeze the ball more tightly.")
    else:
        feedback.append("Great job maintaining a tight squeeze.")

    # Evaluate the position of the thumb
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    thumb_distance_to_squeezing_fingers = min(
        np.linalg.norm(thumb_tip - index_finger_tip),
        np.linalg.norm(thumb_tip - middle_finger_tip)
    )
    
    # Feedback on thumb position
    if thumb_distance_to_squeezing_fingers < 0.1:  # Example threshold for thumb interference
        feedback.append("Keep your thumb away from the squeezing fingers.")
    else:
        feedback.append("Good thumb position away from the squeezing fingers.")

    return feedback

# Function to capture live feed from webcam and provide real-time feedback
def live_feed_feedback():
    cap = cv2.VideoCapture(0)  # Open the webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()  # Read frame from the webcam
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe Hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get feedback for "Side Squeezer"
                feedback = provide_feedback_side_squeezer(hand_landmarks.landmark)
                
                # Display feedback on the frame
                for i, fb in enumerate(feedback):
                    cv2.putText(frame, fb, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the annotated frame
        cv2.imshow('Hand Exercise Feedback - Side Squeezer', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_feed_feedback()
