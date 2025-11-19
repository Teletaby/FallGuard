import cv2
import numpy as np
import math
import torch
import torch.nn.functional as F
import mediapipe as mp 

# Import constants to ensure input sequence shapes match the model
from app.skeleton_lstm import FEATURE_SIZE, SEQUENCE_LENGTH 

# --- Visualization ---
def draw_skeleton(image, results, is_fall_confirmed):
    """Draws the MediaPipe skeleton and connections on the image."""
    
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        
        # Change color to red if fall confirmed
        line_color = (0, 0, 255) if is_fall_confirmed else (255, 0, 0)
        point_color = (0, 255, 255) if is_fall_confirmed else (0, 255, 0)

        # Draw the pose landmarks and connections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=point_color, thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=2)
        )
    return image


# --- Feature Extraction Helpers ---
def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def extract_8_kinematic_features(landmarks):
    """
    Calculates the 8 kinematic features (HWR, TorsoAngle, D, H, FallAngleD, 
    and mock values for P40, HipVx, HipVy).
    """
    mp_pose = mp.solutions.pose
    
    if not landmarks or not landmarks.landmark:
        # Return a zero vector of 8 elements if no landmarks are found
        return np.zeros(8, dtype=np.float32)

    # MediaPipe Landmark Points
    L_SHOULDER = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    R_SHOULDER = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    L_HIP = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
    R_HIP = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
    NOSE = landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]

    # --- 0. HWR (Height-to-Width Ratio of Bounding Box) ---
    x_coords = [lm.x for lm in landmarks.landmark if lm.visibility > 0.5]
    y_coords = [lm.y for lm in landmarks.landmark if lm.visibility > 0.5]
    
    if not x_coords or not y_coords:
        return np.zeros(8, dtype=np.float32)

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    W = max_x - min_x
    H = max_y - min_y
    HWR = H / W if W > 0 else 0.0

    # --- 1. Torso Angle (Angle of the main body axis to the vertical) ---
    shoulder_center_x = (L_SHOULDER.x + R_SHOULDER.x) / 2
    shoulder_center_y = (L_SHOULDER.y + R_SHOULDER.y) / 2
    hip_center_x = (L_HIP.x + R_HIP.x) / 2
    hip_center_y = (L_HIP.y + R_HIP.y) / 2
    
    torso_x_diff = hip_center_x - shoulder_center_x
    torso_y_diff = shoulder_center_y - hip_center_y # Vertical difference
    
    # Angle relative to vertical axis (0 degrees is perfectly upright)
    TorsoAngle = np.degrees(np.arctan2(abs(torso_x_diff), abs(torso_y_diff))) if torso_y_diff != 0 else 0.0

    # --- 2. D (Difference in y-coordinates of head and hip centers) ---
    D = NOSE.y - hip_center_y

    # --- 3. P40 (Average joint velocity) & 4. Hip Vx & 7. Hip Vy --- 
    # These velocity/sequence features are set to 0.0 here as they require sequential data
    P40 = 0.0 
    HipVx = 0.0 
    HipVy = 0.0 
    
    # --- 5. Height of Hip Center (H) ---
    # Normalized height relative to the frame (0.0 is ceiling, 1.0 is floor)
    H = hip_center_y 

    # --- 6. Fall Angle D (Angle of body to horizontal, 90 degrees is vertical) ---
    # Using the same angle calculation as Torso Angle, but relative to horizontal
    FallAngleD = abs(90.0 - TorsoAngle)

    # Create the 8-feature vector
    features_8 = np.array([HWR, TorsoAngle, D, P40, HipVx, H, FallAngleD, HipVy], dtype=np.float32)

    return features_8

def extract_55_features(image, mp_pose_instance):
    """
    Runs MediaPipe, calculates the 8 kinematic features, and pads the result to 55 features.
    """
    mp_pose = mp.solutions.pose
    
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_pose_instance.process(image_rgb)
    image.flags.writeable = True
    
    bbox = None
    feature_vec_55 = np.zeros(FEATURE_SIZE, dtype=np.float32)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks
        
        # 1. Calculate the 8 kinematic features
        features_8 = extract_8_kinematic_features(landmarks)
        
        # 2. Pad to 55 features (to match the model's expected input size)
        feature_vec_55[:8] = features_8
        
        # 3. Calculate Bounding Box (for visualization)
        h, w, _ = image.shape
        x_coords = [lm.x * w for lm in landmarks.landmark if lm.visibility > 0.5]
        y_coords = [lm.y * h for lm in landmarks.landmark if lm.visibility > 0.5]
        
        if x_coords and y_coords:
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            x = int(min_x)
            y = int(min_y)
            bw = int(max_x - min_x)
            bh = int(max_y - min_y)
            bbox = (x, y, bw, bh)
        
    return bbox, feature_vec_55, results

# --- Prediction Utility (CRITICAL FIX APPLIED HERE) ---
def predict_torch(model, sequence_tensor, threshold=0.5):
    """
    Runs inference on the PyTorch model for a single sequence.
    
    Args:
        model (LSTMModel): The loaded PyTorch model.
        sequence_tensor (torch.Tensor): Input tensor of shape (1, SEQUENCE_LENGTH, FEATURE_SIZE).
        threshold (float): Probability threshold for fall classification.
        
    Returns:
        tuple: (prediction_class, probability_of_fall)
    """
    if model is None:
        return 0, 0.0
        
    try:
        with torch.no_grad():
            # Get the raw logits from the model (output shape: (1, 2))
            logits = model(sequence_tensor)
            
            # CRITICAL FIX: Apply Softmax to convert raw logits into probabilities
            # The model is outputting 2 classes, so Softmax is required.
            probs = F.softmax(logits, dim=1) 
            
            # Get the probability of a fall (which is class index 1)
            prob_fall = probs[0, 1].item()
            
            # Classify based on the threshold
            prediction = 1 if prob_fall >= threshold else 0
            
            return prediction, prob_fall
            
    except Exception as e:
        print(f"Error during torch prediction: {e}")
        return 0, 0.0