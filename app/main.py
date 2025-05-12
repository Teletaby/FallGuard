import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import threading
import torch
import cv2
import numpy as np
import requests
import time
from tkinter import Tk, Button, Label, Frame, Scale, HORIZONTAL
from tkinter.filedialog import askopenfilename

# Pushover Notification Function
PUSHOVER_USER_KEY = "uihsvn6ad3btw6ghmq68n5uhzxbmv1"
PUSHOVER_API_TOKEN = "ajrssttu8opfvit5xgh229qrhcv1ew"

def send_pushover_notification(message):
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "message": message
        })
        print("Pushover notification sent.")
    except Exception as e:
        print(f"Failed to send Pushover notification: {e}")

# Your existing import statements
from models.skeleton_lstm import LSTMModel
from utils.video_utils import extract_features_from_frame, draw_pose_on_frame

class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fall Detection System")
        self.root.geometry("500x400")
        
        # Load model configuration
        self.config = self.load_model_config()
        
        # Initialize MediaPipe
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Load the model
        self.model = self.load_model()
        
        # Set up detection parameters
        self.detection_threshold = 0.8  # Increased from 0.5
        self.min_consecutive_detections = 3  # Require multiple consecutive detections
        self.fall_counter = 0
        self.confidence_history = []
        self.smoothing_window = 5  # Number of frames to average for smoothing
        self.fall_confirmation_time = 60  # ~3 seconds at 30 FPS
        
        # Feature debugging
        self.debug_mode = False
        
        # Create UI
        self.create_ui()
        
        # Flag to control camera streaming
        self.is_streaming = False
        self.stream_thread = None
        
        # Initialize cooldown for notifications
        self.last_notification_time = 0  # Track the time of the last notification
        self.notification_cooldown = 60  # Cooldown period (in seconds)
    
    def load_model_config(self):
        """Load model configuration from the config file"""
        config = {
            'input_size': 55,  # Default: 54 keypoints + aspect ratio
            'hidden_size': 128,  # Changed from 1024 to 128
            'output_size': 1, 
            'num_layers': 2,
            'sequence_length': 10
        }
        
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  'models', 'model_config.txt')
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        key, value = line.strip().split('=')
                        if key in ['input_size', 'hidden_size', 'output_size', 'num_layers', 'sequence_length']:
                            config[key] = int(value)
            except Exception as e:
                print(f"Error reading config file: {e}")
        else:
            print(f"Config file not found at {config_path}, using default values")
            
        print(f"Model configuration loaded: {config}")
        return config
    
    def load_model(self):
        # Load the trained model
        try:
            # Use the configuration values instead of hardcoding
            model = LSTMModel(
                input_size=self.config['input_size'], 
                hidden_size=self.config['hidden_size'],
                output_size=self.config['output_size'], 
                num_layers=self.config['num_layers']
            )
            
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'models', 'skeleton_lstm_pytorch_model.pth')
            
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()  # Set to evaluation mode
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def create_ui(self):
        frame = Frame(self.root, padx=20, pady=20)
        frame.pack(expand=True, fill="both")
        
        Label(frame, text="Fall Detection System", font=("Arial", 16)).pack(pady=10)
        
        Button(frame, text="Upload Video", command=self.upload_video, width=20).pack(pady=5)
        Button(frame, text="Start Camera Stream", command=self.toggle_camera_stream, width=20).pack(pady=5)
        
        # Add threshold slider
        threshold_frame = Frame(frame)
        threshold_frame.pack(pady=5, fill="x")
        Label(threshold_frame, text="Detection Threshold:").pack(side="left")
        self.threshold_slider = Scale(threshold_frame, from_=0.1, to=0.9, resolution=0.05, 
                                     orient=HORIZONTAL, command=self.update_threshold)
        self.threshold_slider.set(self.detection_threshold)
        self.threshold_slider.pack(side="right", fill="x", expand=True)
        
        # Debug mode checkbox
        debug_frame = Frame(frame)
        debug_frame.pack(pady=5)
        self.debug_button = Button(debug_frame, text="Toggle Debug Mode", command=self.toggle_debug)
        self.debug_button.pack(side="left", padx=5)
        
        Button(frame, text="Exit", command=self.root.destroy, width=20).pack(pady=10)
        
        self.status_label = Label(frame, text="Ready", font=("Arial", 10))
        self.status_label.pack(pady=10)
        
        # Add current threshold display
        self.threshold_label = Label(frame, text=f"Current threshold: {self.detection_threshold}", font=("Arial", 10))
        self.threshold_label.pack(pady=5)
    
    def update_threshold(self, value):
        self.detection_threshold = float(value)
        self.threshold_label.config(text=f"Current threshold: {self.detection_threshold}")
    
    def toggle_debug(self):
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            self.debug_button.config(text="Debug Mode: ON")
        else:
            self.debug_button.config(text="Debug Mode: OFF")
    
    def upload_video_and_process(self, video_path):
        """Process the video file and return predictions"""
        if not self.model:
            self.status_label.config(text="Model not loaded!")
            return
        
        self.status_label.config(text=f"Processing {os.path.basename(video_path)}...")
        self.root.update()
        
        # Process the video
        try:
            predictions = self.process_video(video_path)
            
            # Count fall detections
            fall_count = sum(1 for pred in predictions if pred == "Fall Detected")
            
            self.status_label.config(text=f"Falls detected: {fall_count}")
            print(f"Processed {len(predictions)} frames, detected {fall_count} falls")
        except Exception as e:
            print(f"Error processing video: {e}")
            self.status_label.config(text=f"Error: {str(e)}")
    
    def process_video(self, video_path):
        """Process video frames and return predictions"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
            
        predictions = []
        sequence_length = self.config['sequence_length']
        feature_buffer = []
        consecutive_fall_detections = 0
        fall_frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract features
            features = extract_features_from_frame(frame, self.pose)
            
            feature_buffer.append(features)
            
            # Once we have enough frames in the sequence
            if len(feature_buffer) >= sequence_length:
                if len(feature_buffer) > sequence_length:
                    feature_buffer = feature_buffer[-sequence_length:]
                
                sequence_array = np.array(feature_buffer, dtype=np.float32)
                sequence_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.model(sequence_tensor)
                    prob = torch.sigmoid(output).item()
                    
                    self.confidence_history.append(prob)
                    if len(self.confidence_history) > self.smoothing_window:
                        self.confidence_history.pop(0)
                    
                    smoothed_prob = sum(self.confidence_history) / len(self.confidence_history)
                    
                    is_fall = smoothed_prob > self.detection_threshold
                    
                    if is_fall:
                        fall_frame_count += 1
                    else:
                        fall_frame_count = 0
                    
                    final_is_fall = fall_frame_count >= self.fall_confirmation_time
                    
                    prediction = "Fall Detected" if final_is_fall else "No Fall Detected"
                
                predictions.append(prediction)
                
                # Send notification if fall is confirmed and cooldown has passed
                if final_is_fall:
                    current_time = time.time()
                    if current_time - self.last_notification_time > self.notification_cooldown:
                        send_pushover_notification("Fall detected! Please check the person.")
                        self.last_notification_time = current_time
                
                # Process image with MediaPipe to get pose landmarks
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    annotated_frame = draw_pose_on_frame(frame, results)
                else:
                    annotated_frame = frame.copy()
                
                color = (0, 0, 255) if final_is_fall else (0, 255, 0)
                cv2.putText(annotated_frame, f"{prediction} ({smoothed_prob:.2f})", (30, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(annotated_frame, f"Threshold: {self.detection_threshold}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if self.debug_mode:
                    cv2.putText(annotated_frame, f"Raw: {prob:.2f}, Smoothed: {smoothed_prob:.2f}", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Fall Detection", annotated_frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return predictions
    
    def upload_video(self):
        """Open file dialog to select a video"""
        video_path = askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if not video_path:
            self.status_label.config(text="No file selected")
            return
        
        threading.Thread(target=self.upload_video_and_process, args=(video_path,), daemon=True).start()
    
    def toggle_camera_stream(self):
        """Start or stop the camera stream"""
        if not self.is_streaming:
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self.camera_prediction_loop, daemon=True)
            self.stream_thread.start()
        else:
            self.is_streaming = False

if __name__ == "__main__":
    root = Tk()
    app = FallDetectionApp(root)
    root.mainloop()
