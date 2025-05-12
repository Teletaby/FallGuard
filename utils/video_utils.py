import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Union

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_features_from_frame(frame, pose=None):
    """
    Extract pose features from a video frame using MediaPipe
    
    Args:
        frame: Video frame (numpy array)
        pose: Optional pre-initialized MediaPipe pose detector
        
    Returns:
        Array of 55 features (54 keypoints + aspect ratio)
    """
    # If pose detector not provided, create a temporary one
    if pose is None:
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose_detector:
            return _extract_pose_features(frame, pose_detector)
    else:
        return _extract_pose_features(frame, pose)

def _extract_pose_features(frame, pose_detector):
    """Helper function to extract pose features"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get frame dimensions for aspect ratio
    height, width, _ = frame.shape
    aspect_ratio = width / height
    
    # Process the frame with MediaPipe
    results = pose_detector.process(frame_rgb)
    
    # Extract relevant pose features
    features = []
    
    if results.pose_landmarks:
        # Create a list of landmark coordinates
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y))
        
        # Extract x and y coordinates for each landmark (skip z)
        for landmark in landmarks:
            features.append(landmark[0])  # x coordinate
            features.append(landmark[1])  # y coordinate
            
        # Calculate additional features that may help with fall detection
        
        # 1. Height of the pose (distance from head to feet)
        head = landmarks[0]  # Nose landmark
        left_foot = landmarks[27]  # Left ankle
        right_foot = landmarks[28]  # Right ankle
        
        # Use the foot that's lower in the frame
        foot = left_foot if left_foot[1] > right_foot[1] else right_foot
        height_ratio = abs(head[1] - foot[1])
        
        # 2. Width of the pose (distance between shoulders)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        width_ratio = abs(left_shoulder[0] - right_shoulder[0])
        
        # 3. Vertical velocity estimation using consecutive frames
        # (This would require tracking across frames, simplified for now)
        # We'll use the relative position of head to center of frame as a proxy
        head_pos = head[1] - 0.5  # Position relative to center of frame
    else:
        # If no landmarks detected, use zeros
        features = [0.0] * 66  # 33 landmarks × 2 coords = 66
        height_ratio = 0
        width_ratio = 0
        head_pos = 0
    
    # Ensure we have exactly 54 features for pose keypoints
    if len(features) > 54:
        features = features[:54]  # Trim excess features
    elif len(features) < 54:
        features.extend([0.0] * (54 - len(features)))  # Pad with zeros
    
    # Add aspect ratio as the 55th feature
    features.append(aspect_ratio)
    
    return np.array(features, dtype=np.float32)

def draw_pose_on_frame(frame, pose_results):
    """
    Draw pose landmarks on a frame
    
    Args:
        frame: Video frame
        pose_results: MediaPipe pose results
        
    Returns:
        Frame with pose landmarks drawn on it
    """
    annotated_frame = frame.copy()
    if pose_results.pose_landmarks:
        # Draw the pose landmarks
        mp_drawing.draw_landmarks(
            annotated_frame, 
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        # Get frame dimensions for calculating position
        height, width, _ = frame.shape
        
        # Draw bounding box around the person
        landmarks = pose_results.pose_landmarks.landmark
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        # Filter out landmarks with low visibility or those that are outside the frame
        valid_x = [x for x in x_coords if 0 <= x <= 1]
        valid_y = [y for y in y_coords if 0 <= y <= 1]
        
        if valid_x and valid_y:
            # Get the bounding box coordinates
            x_min = int(min(valid_x) * width)
            y_min = int(min(valid_y) * height)
            x_max = int(max(valid_x) * width)
            y_max = int(max(valid_y) * height)
            
            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # Calculate and display aspect ratio of the bounding box
            box_width = x_max - x_min
            box_height = y_max - y_min
            aspect_ratio = box_width / box_height if box_height > 0 else 0
            
            cv2.putText(annotated_frame, f"Aspect Ratio: {aspect_ratio:.2f}", 
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return annotated_frame

def stream_camera(pose_detector, callback=None):
    """
    Stream from webcam and process frames in real time
    
    Args:
        pose_detector: MediaPipe pose detector
        callback: Function to call with extracted features
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break
            
            # Extract pose features
            features = extract_features_from_frame(frame, pose_detector)
            
            # Process the frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(frame_rgb)
            
            # Draw pose landmarks
            annotated_frame = draw_pose_on_frame(frame, results)
            
            # Call callback function if provided
            if callback:
                prediction = callback(features)
                # Draw prediction on frame
                cv2.putText(annotated_frame, prediction, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('MediaPipe Pose', annotated_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()