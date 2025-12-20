"""
Exercise-specific feedback functions for hand exercise detection system.

This module contains feedback functions for each supported hand exercise,
providing real-time guidance based on hand landmark analysis.
"""

import numpy as np
from typing import List


# Global variables for finger tip positions
thumb_tip = None
index_finger_tip = None
middle_finger_tip = None
ring_finger_tip = None
pinky_finger_tip = None
thumb_ip = None


def update_finger_tips(landmarks):
    """
    Update global finger tip positions from hand landmarks.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
    """
    global thumb_tip, thumb_ip, index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_finger_tip
    
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_finger_tip = np.array([landmarks[8].x, landmarks[8].y])
    middle_finger_tip = np.array([landmarks[12].x, landmarks[12].y])
    ring_finger_tip = np.array([landmarks[16].x, landmarks[16].y])
    pinky_finger_tip = np.array([landmarks[20].x, landmarks[20].y])
    thumb_ip = np.array([landmarks[3].x, landmarks[3].y])


def default_feedback(landmarks) -> List[str]:
    """
    Default feedback when exercise-specific feedback is not available.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        List of feedback messages
    """
    return ["Feedback is not available for this exercise."]


def provide_feedback_Ball_Grip_Wrist_Down(landmarks) -> List[str]:
    """
    Provide feedback for Ball Grip exercise with wrist down position.
    
    Analyzes grip strength, thumb position, and finger spacing.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        List of feedback messages
    """
    feedback = []
    update_finger_tips(landmarks)
    
    # Get MCP joints for index and middle fingers
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])
    
    # Analyze grip strength
    distance_index_tip_to_mcp = np.linalg.norm(index_finger_tip - index_finger_mcp)
    distance_middle_tip_to_mcp = np.linalg.norm(middle_finger_tip - middle_finger_mcp)
    
    if distance_index_tip_to_mcp < 0.055 and distance_middle_tip_to_mcp < 0.055:
        feedback.append("Release the ball slowly.")
    elif distance_index_tip_to_mcp > 0.06 and distance_middle_tip_to_mcp > 0.06:
        feedback.append("Squeeze the ball tightly.")
    else:
        feedback.append("Maintain your grip.")

    # Analyze thumb position
    distance_thumb_to_index_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_thumb_to_middle_tip = np.linalg.norm(thumb_tip - middle_finger_tip)

    if distance_thumb_to_index_tip < 0.05 and distance_thumb_to_middle_tip < 0.05:
        feedback.append("Good thumb position for a strong grip.")
    else:
        feedback.append("Adjust your thumb position for a better grip.")

    # Analyze finger spacing
    distance_index_to_middle_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_middle_to_ring_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_ring_to_pinky_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # Index and middle finger feedback
    if distance_index_to_middle_tip < 0.02:
        feedback.append("Index and middle fingers are too close.")
    elif distance_index_to_middle_tip > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    # Middle and ring finger feedback
    if distance_middle_to_ring_tip < 0.02:
        feedback.append("Middle and ring fingers are too close.")
    elif distance_middle_to_ring_tip > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    # Ring and pinky finger feedback
    if distance_ring_to_pinky_tip < 0.02:
        feedback.append("Ring and pinky fingers are too close.")
    elif distance_ring_to_pinky_tip > 0.05:
        feedback.append("Ring and pinky fingers are too far apart.")
    else:
        feedback.append("Ring and pinky fingers are correctly positioned.")

    return feedback


def provide_feedback_Ball_Grip_Wrist_UP(landmarks) -> List[str]:
    """
    Provide feedback for Ball Grip exercise with wrist up position.
    
    Similar to wrist down but with adjusted thresholds for wrist-up position.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        List of feedback messages
    """
    feedback = []
    update_finger_tips(landmarks)
    
    # Get MCP joints for index and middle fingers
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])

    # Analyze grip strength
    distance_index_tip_to_mcp = np.linalg.norm(index_finger_tip - index_finger_mcp)
    distance_middle_tip_to_mcp = np.linalg.norm(middle_finger_tip - middle_finger_mcp)

    if distance_index_tip_to_mcp < 0.055 and distance_middle_tip_to_mcp < 0.055:
        feedback.append("Release the ball slowly.")
    elif distance_index_tip_to_mcp > 0.06 and distance_middle_tip_to_mcp > 0.06:
        feedback.append("Squeeze the ball tightly.")
    else:
        feedback.append("Maintain your grip.")

    # Analyze thumb position
    distance_thumb_to_index_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_thumb_to_middle_tip = np.linalg.norm(thumb_tip - middle_finger_tip)

    if distance_thumb_to_index_tip < 0.05 and distance_thumb_to_middle_tip < 0.05:
        feedback.append("Good thumb position for a strong grip.")
    else:
        feedback.append("Adjust your thumb position for a better grip.")

    # Analyze finger spacing
    distance_index_to_middle_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_middle_to_ring_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_ring_to_pinky_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # Index and middle finger feedback
    if distance_index_to_middle_tip < 0.02:
        feedback.append("Index and middle fingers are too close.")
    elif distance_index_to_middle_tip > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    # Middle and ring finger feedback
    if distance_middle_to_ring_tip < 0.02:
        feedback.append("Middle and ring fingers are too close.")
    elif distance_middle_to_ring_tip > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    # Ring and pinky finger feedback
    if distance_ring_to_pinky_tip < 0.02:
        feedback.append("Ring and pinky fingers are too close.")
    elif distance_ring_to_pinky_tip > 0.05:
        feedback.append("Ring and pinky fingers are too far apart.")
    else:
        feedback.append("Ring and pinky fingers are correctly positioned.")

    return feedback


def provide_feedback_Pinch(landmarks) -> List[str]:
    """
    Provide feedback for Pinch exercise.
    
    Analyzes pinch distance between thumb and index finger.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        List of feedback messages
    """
    feedback = []
    update_finger_tips(landmarks)
    
    # Analyze pinch distance
    pinch_distance = np.linalg.norm(thumb_tip - index_finger_tip)

    if pinch_distance > 0.17:
        feedback.append("Try to bring your thumb and index finger closer.")
    else:
        feedback.append("Good pinch! Maintain the grip.")

    # Analyze finger positions relative to neighbors
    index_to_middle_distance = np.linalg.norm(index_finger_tip - middle_finger_tip)
    middle_to_ring_distance = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    ring_to_pinky_distance = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # Index and middle finger feedback
    if index_to_middle_distance < 0.01:
        feedback.append("Index and middle fingers are too close.")
    elif index_to_middle_distance > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    # Middle and ring finger feedback
    if middle_to_ring_distance < 0.01:
        feedback.append("Middle and ring fingers are too close.")
    elif middle_to_ring_distance > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    # Ring and pinky finger feedback
    if ring_to_pinky_distance < 0.01:
        feedback.append("Ring and pinky fingers are too close.")
    elif ring_to_pinky_distance > 0.07:
        feedback.append("Ring and pinky fingers are too far apart.")
    else:
        feedback.append("Ring and pinky fingers are correctly positioned.")

    return feedback


def provide_feedback_Thumb_Extend(landmarks) -> List[str]:
    """
    Provide feedback for Thumb Extend exercise.
    
    Analyzes thumb extension relative to other finger bases.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        List of feedback messages
    """
    feedback = []
    update_finger_tips(landmarks)
    
    # Get MCP joints for index, middle, and ring fingers
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])
    ring_finger_mcp = np.array([landmarks[13].x, landmarks[13].y])

    # Calculate distances from thumb tip to MCPs
    thumb_tip_to_index_mcp_distance = np.linalg.norm(thumb_tip - index_finger_mcp)
    thumb_tip_to_middle_mcp_distance = np.linalg.norm(thumb_tip - middle_finger_mcp)
    thumb_tip_to_ring_mcp_distance = np.linalg.norm(thumb_tip - ring_finger_mcp)
    
    # Calculate distances from thumb IP to MCPs
    thumb_ip_to_index_mcp_distance = np.linalg.norm(thumb_ip - index_finger_mcp)
    thumb_ip_to_middle_mcp_distance = np.linalg.norm(thumb_ip - middle_finger_mcp)
    thumb_ip_to_ring_mcp_distance = np.linalg.norm(thumb_ip - ring_finger_mcp)
    
    # Thumb IP to index MCP feedback
    if thumb_ip_to_index_mcp_distance > 0.07:
        feedback.append("Thumb center is far from index finger base; try to keep it closer by squeezing tighter.")
    else:
        feedback.append("Good distance maintained between thumb center and base of index finger.")

    # Thumb IP to middle MCP feedback
    if thumb_ip_to_middle_mcp_distance >= 0.065:
        feedback.append("Thumb center is far from the middle finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the middle finger base.")

    # Thumb IP to ring MCP feedback
    if thumb_ip_to_ring_mcp_distance >= 0.095:
        feedback.append("Thumb center is far from the ring finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the ring finger base.")

    # Thumb tip to index MCP feedback
    if thumb_tip_to_index_mcp_distance > 0.085:
        feedback.append("Thumb tip is too far from index finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the index finger base.")

    # Thumb tip to middle MCP feedback
    if thumb_tip_to_middle_mcp_distance > 0.08:
        feedback.append("Thumb tip is too far from middle finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the middle finger base.")

    # Thumb tip to ring MCP feedback
    if thumb_tip_to_ring_mcp_distance > 0.064:
        feedback.append("Thumb tip is too far from ring finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the ring finger base.")
        
    return feedback


def provide_feedback_Opposition(landmarks) -> List[str]:
    """
    Provide feedback for Opposition exercise.
    
    Analyzes thumb opposition to other fingers with specific thresholds.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        List of feedback messages
    """
    feedback = []
    update_finger_tips(landmarks)
    
    # Get MCP joints
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])
    ring_finger_mcp = np.array([landmarks[13].x, landmarks[13].y])
    
    # Calculate distances
    thumb_tip_to_index_mcp_distance = np.linalg.norm(thumb_tip - index_finger_mcp)
    thumb_tip_to_middle_mcp_distance = np.linalg.norm(thumb_tip - middle_finger_mcp)
    thumb_tip_to_ring_mcp_distance = np.linalg.norm(thumb_tip - ring_finger_mcp)
    thumb_ip_to_index_mcp_distance = np.linalg.norm(thumb_ip - index_finger_mcp)
    thumb_ip_to_middle_mcp_distance = np.linalg.norm(thumb_ip - middle_finger_mcp)
    thumb_ip_to_ring_mcp_distance = np.linalg.norm(thumb_ip - ring_finger_mcp)
    
    # Thumb IP feedback
    if thumb_ip_to_index_mcp_distance > 0.095:
        feedback.append("Thumb center is far from index finger base; try to keep it closer by squeezing tighter.")
    else:
        feedback.append("Good distance maintained between thumb center and base of index finger.")
        
    if thumb_ip_to_middle_mcp_distance >= 0.06:
        feedback.append("Thumb center is far from the middle finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the middle finger base.")
        
    if thumb_ip_to_ring_mcp_distance >= 0.045:
        feedback.append("Thumb center is far from the ring finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the ring finger base.")
    
    # Thumb tip feedback
    if thumb_tip_to_index_mcp_distance > 0.1:
        feedback.append("Thumb tip is too far from index finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the index finger base.")
        
    if thumb_tip_to_middle_mcp_distance > 0.09:
        feedback.append("Thumb tip is too far from middle finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the middle finger base.")
        
    if thumb_tip_to_ring_mcp_distance > 0.11:
        feedback.append("Thumb tip is too far from ring finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the ring finger base.")
        
    return feedback


def provide_feedback_Extend_Out(landmarks) -> List[str]:
    """
    Provide feedback for Extend Out exercise.
    
    Analyzes finger extension and positioning.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        List of feedback messages
    """
    feedback = []
    update_finger_tips(landmarks)
    
    # Get specific landmarks
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    ring_finger_dip = np.array([landmarks[15].x, landmarks[15].y])
    
    # Calculate distances
    distance_between_index_tip_and_middle_finger_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_between_middle_tip_and_ring_finger_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_between_thumb_tip_and_index_finger_mcp = np.linalg.norm(thumb_tip - index_finger_mcp)
    distance_between_pinky_finger_tip_and_ring_finger_dip = np.linalg.norm(ring_finger_dip - pinky_finger_tip)
    
    # Index and middle finger feedback
    if distance_between_index_tip_and_middle_finger_tip >= 0.05:
        feedback.append("Keep index finger and middle finger attached with each other!")
    else:
        feedback.append("Index finger and middle finger are properly attached.")
        
    # Middle and ring finger feedback
    if distance_between_middle_tip_and_ring_finger_tip >= 0.07:
        feedback.append("Keep middle finger and ring finger attached with each other!")
    else:
        feedback.append("Middle finger and ring finger are properly attached.")
        
    # Thumb position feedback
    if distance_between_thumb_tip_and_index_finger_mcp <= 0.06:
        feedback.append("Keep thumb and index finger base far from each other!")
    elif distance_between_thumb_tip_and_index_finger_mcp >= 0.061 and distance_between_thumb_tip_and_index_finger_mcp <= 0.15:
        feedback.append("Good distance maintenance for thumb.")
    else:
        feedback.append("Thumb is very far from index finger base so bend it and keep close!")
        
    # Pinky finger feedback
    if distance_between_pinky_finger_tip_and_ring_finger_dip <= 0.08:
        feedback.append("Keep ring finger upper joint and pinky finger far from each other!")
    elif distance_between_pinky_finger_tip_and_ring_finger_dip > 0.081 and distance_between_pinky_finger_tip_and_ring_finger_dip <= 0.14:
        feedback.append("Good distance maintenance for pinky finger.")
    else:
        feedback.append("Pinky finger is very far from ring finger keep it close!")
        
    return feedback


def provide_feedback_Finger_Bend(landmarks) -> List[str]:
    """
    Provide feedback for Finger Bend exercise.
    
    Analyzes finger bending and thumb positioning.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        List of feedback messages
    """
    feedback = []
    update_finger_tips(landmarks)
    
    # Calculate distances between all fingertips
    distance_between_index_tip_and_middle_finger_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_between_middle_tip_and_ring_finger_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_between_ring_tip_and_pinky_finger_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)
    distance_between_thumb_tip_and_index_finger_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_between_thumb_tip_and_middle_finger_tip = np.linalg.norm(thumb_tip - middle_finger_tip)
    distance_between_thumb_tip_and_ring_finger_tip = np.linalg.norm(thumb_tip - ring_finger_tip)
    distance_between_thumb_tip_and_pinky_finger_tip = np.linalg.norm(thumb_tip - pinky_finger_tip)
    
    # Index and middle finger feedback
    if distance_between_index_tip_and_middle_finger_tip >= 0.06:
        feedback.append("Keep index finger and middle finger close to each other!")
    else:
        feedback.append("Index finger and middle finger are properly aligned.")
        
    # Middle and ring finger feedback
    if distance_between_middle_tip_and_ring_finger_tip >= 0.06:
        feedback.append("Keep middle finger and ring finger close to each other!")
    else:
        feedback.append("Middle finger and ring finger are properly aligned.")
        
    # Ring and pinky finger feedback
    if distance_between_ring_tip_and_pinky_finger_tip >= 0.06:
        feedback.append("Keep ring finger and pinky finger close to each other!")
    else:
        feedback.append("Ring finger and pinky finger are properly aligned.")
        
    # Thumb feedback for all fingers
    if distance_between_thumb_tip_and_index_finger_tip >= 0.085:
        feedback.append("Keep index finger and thumb close to each other!")
    else:
        feedback.append("Index finger and thumb are properly aligned.")
        
    if distance_between_thumb_tip_and_middle_finger_tip >= 0.085:
        feedback.append("Keep middle finger and thumb close to each other!")
    else:
        feedback.append("Middle finger and thumb are properly aligned.")
        
    if distance_between_thumb_tip_and_ring_finger_tip >= 0.085:
        feedback.append("Keep ring finger and thumb close to each other!")
    else:
        feedback.append("Ring finger and thumb are properly aligned.")
        
    if distance_between_thumb_tip_and_pinky_finger_tip >= 0.085:
        feedback.append("Keep pinky finger and thumb close to each other!")
    else:
        feedback.append("Pinky finger and thumb are properly aligned.")
        
    return feedback


def provide_feedback_Side_Squzzer(landmarks) -> List[str]:
    """
    Provide feedback for Side Squeezer exercise.
    
    Analyzes squeezing strength and finger positioning.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        List of feedback messages
    """
    feedback = []
    update_finger_tips(landmarks)
    
    # Analyze squeezing distance
    distance_between_tips = np.linalg.norm(index_finger_tip - middle_finger_tip)
    if distance_between_tips > 0.05:
        feedback.append("Try to squeeze the ball more tightly and make the distance min. between index and middle finger.")
    else:
        feedback.append("Great job maintaining a tight squeeze now release and repeat it.")
    
    # Analyze thumb position relative to squeezing fingers
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_finger_pip = np.array([landmarks[6].x, landmarks[6].y])
    thumb_distance_to_squeezing_fingers = min(
        np.linalg.norm(thumb_tip - index_finger_pip),
        np.linalg.norm(thumb_tip - middle_finger_tip)
    )
    
    if thumb_distance_to_squeezing_fingers >= 0.045:
        feedback.append("Keep your thumb attached with squeezing fingers.")
    else:
        feedback.append("Good thumb position with squeezing fingers. Keep them attached.")
    
    # Analyze ring and pinky finger bending
    ring_finger_mcp = np.array([landmarks[13].x, landmarks[13].y])
    pinky_finger_mcp = np.array([landmarks[17].x, landmarks[17].y])
    
    ring_finger_distance_to_ring_finger_mcp = np.linalg.norm(ring_finger_tip - ring_finger_mcp)
    if ring_finger_distance_to_ring_finger_mcp >= 0.04:
        feedback.append("Try to bend your ring finger more inward.")
    else:
        feedback.append("Good bending of ring finger.")
    
    pinky_finger_distance_to_pinky_finger_mcp = np.linalg.norm(pinky_finger_tip - pinky_finger_mcp)
    if pinky_finger_distance_to_pinky_finger_mcp >= 0.04:
        feedback.append("Try to bend your pinky finger more inward.")
    else:
        feedback.append("Good bending of pinky finger.")
        
    return feedback 