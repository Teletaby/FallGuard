import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
# NOTE: MediaPipe and OpenCV imports are commented out for simplicity, 
# as they require external environment setup.
# import cv2 
# import mediapipe as mp 

# --- Model Definition (Duplicated/Centralized for easier import if path issues persist) ---
# NOTE: In a real project, this class should be imported from models.skeleton_lstm
# but is duplicated here temporarily to resolve immediate ModuleNotFoundErrors.
class LSTMModel(nn.Module):
    """A simple LSTM model for sequence classification."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
# -----------------------------------------------------------------------------------


# --- Model and Config Paths (Updated to be relative to the project root) ---
MODEL_DIR = 'models'
MODEL_FILENAME = 'skeleton_lstm_pytorch_model.pth'
CONFIG_FILENAME = 'model_config.json'
# Assuming main.py is in 'app/', so we navigate up one directory ('..') to find 'models/'
MODEL_PATH = os.path.join('..', MODEL_DIR, MODEL_FILENAME)
CONFIG_PATH = os.path.join('..', MODEL_DIR, CONFIG_FILENAME)

class FallDetector:
    def __init__(self):
        self.model = None
        self.config = self._load_model_config()
        self._load_model()
        self.sequence_length = self.config.get('sequence_length', 10)
        self.input_size = self.config.get('input_size', 55)
        self.feature_sequence = [] # Stores the last N frames of features
        self.fall_detected = False
        self.latest_fall_prob = 0.0

        # Initialize MediaPipe (commented out)
        # self.mp_pose = mp.solutions.pose
        # self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        print(f"Detector initialized. Sequence Length: {self.sequence_length}, Input Size: {self.input_size}")

    def _load_model_config(self):
        """Loads model configuration from the saved JSON file."""
        if not os.path.exists(CONFIG_PATH):
            print(f"[ERROR] Configuration file not found at {CONFIG_PATH}. Using default config.")
            return {
                'input_size': 55,
                'hidden_size': 128,
                'output_size': 1,
                'num_layers': 2,
                'sequence_length': 10
            }
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load configuration: {e}")
            return {}

    def _load_model(self):
        """Initializes the model and loads weights from the PTH file."""
        input_size = self.config.get('input_size', 55)
        hidden_size = self.config.get('hidden_size', 128)
        num_layers = self.config.get('num_layers', 2)
        output_size = self.config.get('output_size', 1)

        self.model = LSTMModel(input_size, hidden_size, output_size, num_layers)

        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
                self.model.eval()
                print(f"[INFO] LSTM Model loaded successfully from {MODEL_PATH}")
            except Exception as e:
                print(f"[ERROR] Failed to load model weights from {MODEL_PATH}: {e}")
                self.model = None
        else:
            # THIS IS THE CRITICAL PATH FIX FOR main.py
            print(f"[ERROR] LSTM Model file not found at {MODEL_PATH}. Using mock logic.")
            self.model = None


    def process_frame(self, frame_features):
        """
        Simulates processing a new frame of features.
        In a real application, this would involve MediaPipe processing.
        """
        # Ensure features have the correct shape
        if len(frame_features) != self.input_size:
            print(f"[WARN] Expected {self.input_size} features, got {len(frame_features)}. Skipping frame.")
            return

        # Add new features and maintain sequence length
        self.feature_sequence.append(frame_features)
        if len(self.feature_sequence) > self.sequence_length:
            self.feature_sequence.pop(0)

        fall_probability = 0.0
        
        # Only predict if we have a full sequence
        if len(self.feature_sequence) == self.sequence_length and self.model:
            sequence_np = np.array(self.feature_sequence, dtype=np.float32)
            sequence_tensor = torch.tensor(sequence_np[np.newaxis, :, :], dtype=torch.float32)
            
            with torch.no_grad():
                output = self.model(sequence_tensor)
                fall_probability = torch.sigmoid(output).item()
                
            prediction = 1 if fall_probability > 0.5 else 0
            
            self.latest_fall_prob = fall_probability # Store probability
            
            if prediction == 1:
                if not self.fall_detected:
                    self.fall_detected = True
                    print(f"!!! FALL DETECTED (Prob: {fall_probability:.4f}) !!!")
            else:
                self.fall_detected = False

        return self.fall_detected, fall_probability

# --- Simulation Code ---
def generate_mock_frame_features(is_fall_imminent, feature_size):
    """Generates mock features for testing purposes."""
    features = np.random.rand(feature_size).astype('float32')
    
    # Simulate low height (index 5) for a fall
    if is_fall_imminent:
        features[5] = np.random.uniform(0.05, 0.4) 
        features[0] = np.random.uniform(0.7, 0.9) 
    else:
        features[5] = np.random.uniform(0.6, 0.95)
        features[0] = np.random.uniform(0.1, 0.5)
        
    return features.tolist()

if __name__ == '__main__':
    detector = FallDetector()
    
    # Simple simulation loop
    NUM_FRAMES = 50
    FALL_START_FRAME = 25
    FALL_END_FRAME = 35

    print("\n--- Starting Live Simulation (50 frames) ---")
    
    for i in range(1, NUM_FRAMES + 1):
        is_fall_imminent = FALL_START_FRAME <= i <= FALL_END_FRAME
        
        # Generate mock features for the current frame
        features = generate_mock_frame_features(is_fall_imminent, detector.input_size)
        
        # Process the frame
        is_fall, probability = detector.process_frame(features)
        
        status = "FALLING" if is_fall else "Normal"
        phase = "Fall Phase" if is_fall_imminent else "Normal Phase"
        
        # Print status every few frames for clarity
        if i % 5 == 0 or is_fall or is_fall_imminent:
             print(f"Frame {i}/{NUM_FRAMES} ({phase}): Status={status} | Probability={probability:.4f}")
        
        time.sleep(0.05) # Simulate frame processing time

    print("\n--- Simulation Finished ---")