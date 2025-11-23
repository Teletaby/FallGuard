import time
import logging
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from flask import Flask, request, jsonify, Response, render_template
import os
import threading
from datetime import datetime
from werkzeug.utils import secure_filename
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose components globally
mp_pose = mp.solutions.pose

# ----------------------------------------------------
# MODEL CONFIGURATION
# ----------------------------------------------------

# Model parameters (must match your trained model)
FEATURE_SIZE = 55
HIDDEN_SIZE = 128
OUTPUT_SIZE = 2  # or 1, depending on your model
NUM_LAYERS = 2
SEQUENCE_LENGTH = 30

MODEL_FILE = 'models/skeleton_lstm_pytorch_model.pth'
LSTM_MODEL = None
device = torch.device('cpu')  # Use CPU for cloud deployment

# Try to import your model classes
try:
    from app.skeleton_lstm import LSTMModel
    from app.video_utils import extract_55_features, predict_torch
    MODEL_AVAILABLE = True
    logger.info("Model modules imported successfully")
except ImportError as e:
    logger.warning(f"Could not import model modules: {e}")
    MODEL_AVAILABLE = False
    # Define fallback minimal model
    class LSTMModel(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            out = self.fc(h_n[-1])
            return out

# Load the model
try:
    LSTM_MODEL = LSTMModel(FEATURE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    
    if os.path.exists(MODEL_FILE):
        LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
        LSTM_MODEL.to(device)
        LSTM_MODEL.eval()
        logger.info(f"✓ LSTM Model loaded from {MODEL_FILE}")
    else:
        logger.warning(f"Model file not found: {MODEL_FILE}. Using heuristic detection only.")
        LSTM_MODEL = None
except Exception as e:
    logger.error(f"Failed to load LSTM model: {e}")
    LSTM_MODEL = None

# ----------------------------------------------------
# UTILITY CLASSES
# ----------------------------------------------------

class FallTimer:
    def __init__(self, threshold=3):
        self.start_time = None
        self.threshold = threshold

    def update(self, is_falling):
        current_time = time.time()
        if is_falling:
            if self.start_time is None:
                self.start_time = current_time
            elif current_time - self.start_time >= self.threshold:
                return True
        else:
            self.start_time = None
        return False
        
    def reset(self):
        self.start_time = None

# ----------------------------------------------------
# FEATURE EXTRACTION (FALLBACK)
# ----------------------------------------------------

def extract_features_fallback(landmarks):
    """
    Fallback feature extraction if video_utils is not available.
    Extracts basic geometric features from pose landmarks.
    """
    if not landmarks or len(landmarks) < 33:
        return np.zeros(FEATURE_SIZE, dtype=np.float32)
    
    features = []
    
    # Key landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    NOSE = 0
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    try:
        # Get key points
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        nose = landmarks[NOSE]
        left_ankle = landmarks[LEFT_ANKLE]
        right_ankle = landmarks[RIGHT_ANKLE]
        
        # Shoulder center
        shoulder_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        
        # Hip center
        hip_x = (left_hip['x'] + right_hip['x']) / 2
        hip_y = (left_hip['y'] + right_hip['y']) / 2
        
        # Ankle center
        ankle_y = (left_ankle['y'] + right_ankle['y']) / 2
        
        # Feature 1: Height-Width Ratio (HWR)
        torso_height = abs(shoulder_y - hip_y)
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
        hwr = torso_height / shoulder_width if shoulder_width > 0 else 1.0
        features.append(hwr)
        
        # Feature 2: Torso Angle
        torso_angle = np.degrees(np.arctan2(hip_y - shoulder_y, hip_x - shoulder_x))
        features.append(abs(torso_angle))
        
        # Feature 3: Head-Hip vertical distance
        head_hip_dist = abs(nose['y'] - hip_y)
        features.append(head_hip_dist)
        
        # Feature 4: Body verticality
        body_height = abs(ankle_y - nose['y'])
        features.append(body_height)
        
        # Feature 5: Hip height (normalized)
        features.append(hip_y)
        
        # Feature 6: Shoulder height
        features.append(shoulder_y)
        
        # Feature 7: Fall angle (angle of body from vertical)
        fall_angle = 90 - abs(torso_angle)
        features.append(fall_angle)
        
        # Add all landmark coordinates (33 landmarks * 3 coords = 99 values)
        # But we need exactly FEATURE_SIZE features
        for lm in landmarks:
            if len(features) >= FEATURE_SIZE:
                break
            features.extend([lm['x'], lm['y']])
        
        # Pad with zeros if needed
        while len(features) < FEATURE_SIZE:
            features.append(0.0)
        
        # Truncate if too long
        features = features[:FEATURE_SIZE]
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return np.zeros(FEATURE_SIZE, dtype=np.float32)

# ----------------------------------------------------
# HEURISTIC FALL DETECTION
# ----------------------------------------------------

def detect_fall_heuristic(landmarks):
    """
    Heuristic fall detection based on body geometry.
    Returns (is_falling: bool, confidence: float)
    """
    if not landmarks or len(landmarks) < 33:
        return False, 0.0
    
    try:
        # Key landmarks
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        NOSE = 0
        
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        nose = landmarks[NOSE]
        
        # Calculate features
        shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        hip_y = (left_hip['y'] + right_hip['y']) / 2
        shoulder_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        hip_x = (left_hip['x'] + right_hip['x']) / 2
        
        torso_height = abs(shoulder_y - hip_y)
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
        hwr = torso_height / shoulder_width if shoulder_width > 0 else 1.0
        
        torso_angle = abs(np.degrees(np.arctan2(hip_y - shoulder_y, hip_x - shoulder_x)))
        head_hip_dist = abs(nose['y'] - hip_y)
        
        # Fall detection logic
        fall_score = 0.0
        
        # Criterion 1: Low HWR (person is horizontal)
        if hwr < 0.7:
            fall_score += 0.3
            if hwr < 0.5:
                fall_score += 0.2
        
        # Criterion 2: High torso angle (leaning/horizontal)
        if torso_angle > 45:
            fall_score += 0.25
            if torso_angle > 70:
                fall_score += 0.15
        
        # Criterion 3: Hip is high (person low to ground)
        if hip_y > 0.6:
            fall_score += 0.2
            if hip_y > 0.75:
                fall_score += 0.2
        
        # Criterion 4: Small head-hip distance
        if head_hip_dist < 0.15:
            fall_score += 0.15
        
        return fall_score > 0.5, min(fall_score, 0.99)
    
    except Exception as e:
        logger.error(f"Heuristic detection error: {e}")
        return False, 0.0

# ----------------------------------------------------
# POSE PROCESSOR
# ----------------------------------------------------

class PoseStreamProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Use 0 for faster processing on cloud
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmark_history = {}
        self.history_size = 5
        logger.info("MediaPipe Pose initialized.")

    def _smooth_landmarks(self, camera_index: str, landmarks: list[dict]) -> list[dict]:
        if camera_index not in self.landmark_history:
            self.landmark_history[camera_index] = deque(maxlen=self.history_size)
        
        current_frame = np.array([
            [lm['x'], lm['y'], lm['z'], lm['visibility']] 
            for lm in landmarks
        ])
        
        self.landmark_history[camera_index].append(current_frame)
        
        if len(self.landmark_history[camera_index]) < 2:
            return landmarks
        
        history = list(self.landmark_history[camera_index])
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3]) 
        weights = weights[-len(history):]
        weights = weights / weights.sum()
        
        smoothed = np.zeros_like(current_frame)
        for i, frame in enumerate(history):
            smoothed += frame * weights[i]
        
        smoothed_landmarks = []
        for i in range(len(landmarks)):
            smoothed_landmarks.append({
                'x': float(smoothed[i, 0]),
                'y': float(smoothed[i, 1]),
                'z': float(smoothed[i, 2]),
                'visibility': float(smoothed[i, 3])
            })
        
        return smoothed_landmarks

    def _is_valid_human_pose(self, landmarks: list[dict]) -> bool:
        if not landmarks or len(landmarks) < 33:
            return False
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        key_landmarks = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        visible_key_landmarks = sum(1 for idx in key_landmarks if landmarks[idx].get("visibility", 0) > 0.5)
        
        return visible_key_landmarks >= 2

    def process_frame(self, frame, camera_index: str = "0"):
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                if camera_index in self.landmark_history:
                    self.landmark_history[camera_index].clear()
                return None
            
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": landmark.x, 
                    "y": landmark.y, 
                    "z": landmark.z, 
                    "visibility": landmark.visibility
                })
            
            if not self._is_valid_human_pose(landmarks):
                return None
            
            smoothed_landmarks = self._smooth_landmarks(camera_index, landmarks)
            return {"landmarks": smoothed_landmarks}
        
        except Exception as e:
            logger.error(f"Pose processing error: {e}")
            return None
    
    def reset_camera_history(self, camera_index: str):
        if camera_index in self.landmark_history:
            self.landmark_history[camera_index].clear()

# ----------------------------------------------------
# CAMERA MANAGER
# ----------------------------------------------------

class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.camera_streams = {}
        self.camera_locks = {}
        self.pose_sequences = {}  # Store pose sequences per camera
        self.settings = {
            'fall_threshold': 0.70,
            'fall_delay_seconds': 2
        }
        self.upload_folder = 'uploads'
        os.makedirs(self.upload_folder, exist_ok=True)
        
    def add_camera(self, camera_id: str, name: str, source: str):
        """Add a new camera"""
        self.cameras[camera_id] = {
            'id': camera_id,
            'name': name,
            'source': source,
            'isLive': False,
            'color': 'green',
            'status': 'Stopped',
            'confidence_score': 0.0,
            'model_threshold': self.settings['fall_threshold'],
            'fps': 0
        }
        self.camera_locks[camera_id] = threading.Lock()
        self.pose_sequences[camera_id] = deque(maxlen=SEQUENCE_LENGTH)
        logger.info(f"Camera added: {camera_id} ({name})")
        
    def start_camera(self, camera_id: str):
        """Start streaming from a camera"""
        if camera_id not in self.cameras:
            return False
            
        camera_info = self.cameras[camera_id]
        source = camera_info['source']
        
        try:
            # Check if it's a file path
            if os.path.isfile(source):
                cap = cv2.VideoCapture(source)
            else:
                # Try to parse as integer (webcam index) or URL
                try:
                    source_int = int(source)
                    cap = cv2.VideoCapture(source_int)
                except ValueError:
                    cap = cv2.VideoCapture(source)
        except Exception as e:
            logger.error(f"Error opening camera {camera_id}: {e}")
            return False
        
        if cap.isOpened():
            self.camera_streams[camera_id] = cap
            self.cameras[camera_id]['isLive'] = True
            self.cameras[camera_id]['status'] = 'Active'
            self.cameras[camera_id]['color'] = 'green'
            logger.info(f"Camera started: {camera_id}")
            return True
        else:
            logger.error(f"Failed to open camera: {camera_id}")
            return False
    
    def stop_camera(self, camera_id: str):
        """Stop a camera stream"""
        if camera_id in self.camera_streams:
            self.camera_streams[camera_id].release()
            del self.camera_streams[camera_id]
            self.cameras[camera_id]['isLive'] = False
            self.cameras[camera_id]['status'] = 'Stopped'
            self.cameras[camera_id]['color'] = 'gray'
            # Clear pose sequence
            if camera_id in self.pose_sequences:
                self.pose_sequences[camera_id].clear()
            logger.info(f"Camera stopped: {camera_id}")
    
    def get_frame(self, camera_id: str):
        """Get the latest frame from a camera"""
        if camera_id not in self.camera_streams:
            return None
            
        with self.camera_locks[camera_id]:
            cap = self.camera_streams[camera_id]
            ret, frame = cap.read()
            if ret:
                return frame
            else:
                # Video ended, loop it
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    return frame
            return None
    
    def remove_camera(self, camera_id: str):
        """Remove a camera completely"""
        self.stop_camera(camera_id)
        if camera_id in self.cameras:
            # Delete video file if it exists
            source = self.cameras[camera_id]['source']
            if os.path.isfile(source) and source.startswith(self.upload_folder):
                try:
                    os.remove(source)
                    logger.info(f"Deleted video file: {source}")
                except Exception as e:
                    logger.error(f"Error deleting file: {e}")
            del self.cameras[camera_id]
        if camera_id in self.camera_locks:
            del self.camera_locks[camera_id]
        if camera_id in self.pose_sequences:
            del self.pose_sequences[camera_id]
        logger.info(f"Camera removed: {camera_id}")

# ----------------------------------------------------
# FLASK APPLICATION
# ----------------------------------------------------

# Set template folder correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.secret_key = os.environ.get('FALLGUARD_SECRET', 'fallguard_secret_key_2024')

# Initialize components
pose_processor = PoseStreamProcessor()
camera_manager = CameraManager()
camera_timers = {}

# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        data = request.get_json()
        camera_manager.settings['fall_threshold'] = float(data.get('fall_threshold', 0.70))
        camera_manager.settings['fall_delay_seconds'] = int(data.get('fall_delay_seconds', 2))
        
        # Update all camera thresholds
        for cam_id in camera_manager.cameras:
            camera_manager.cameras[cam_id]['model_threshold'] = camera_manager.settings['fall_threshold']
        
        # Update fall timers
        for timer in camera_timers.values():
            timer.threshold = camera_manager.settings['fall_delay_seconds']
        
        logger.info(f"Settings updated: threshold={camera_manager.settings['fall_threshold']}, delay={camera_manager.settings['fall_delay_seconds']}s")
        
        return jsonify({'success': True, 'settings': camera_manager.settings, 'message': 'Settings updated'})
    
    return jsonify({'success': True, 'settings': camera_manager.settings})

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    cameras = list(camera_manager.cameras.values())
    return jsonify({'success': True, 'cameras': cameras})

@app.route('/api/cameras/all_definitions', methods=['GET'])
def get_all_definitions():
    definitions = list(camera_manager.cameras.values())
    return jsonify({'success': True, 'definitions': definitions})

@app.route('/api/cameras/add', methods=['POST'])
def add_camera():
    data = request.get_json()
    name = data.get('name')
    source = data.get('source')
    
    if not name or source is None:
        return jsonify({'success': False, 'message': 'Name and source required'}), 400
    
    camera_id = f"cam_{int(time.time() * 1000)}"
    camera_manager.add_camera(camera_id, name, str(source))
    camera_manager.start_camera(camera_id)
    
    return jsonify({'success': True, 'camera_id': camera_id, 'message': f'Camera {name} added'})

@app.route('/api/cameras/add_existing', methods=['POST'])
def add_existing_camera():
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if not camera_id:
        return jsonify({'success': False, 'message': 'Camera ID required'}), 400
    
    if camera_id in camera_manager.cameras:
        success = camera_manager.start_camera(camera_id)
        if success:
            return jsonify({'success': True, 'message': 'Camera started'})
        else:
            return jsonify({'success': False, 'message': 'Failed to start camera'}), 500
    
    return jsonify({'success': False, 'message': 'Camera not found'}), 404

@app.route('/api/cameras/stop/<camera_id>', methods=['POST'])
def stop_camera(camera_id):
    if camera_id not in camera_manager.cameras:
        return jsonify({'success': False, 'message': 'Camera not found'}), 404
    
    camera_manager.stop_camera(camera_id)
    if camera_id in camera_timers:
        camera_timers[camera_id].reset()
    
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/api/cameras/remove/<camera_id>', methods=['DELETE'])
def remove_camera(camera_id):
    if camera_id not in camera_manager.cameras:
        return jsonify({'success': False, 'message': 'Camera not found'}), 404
    
    camera_manager.remove_camera(camera_id)
    if camera_id in camera_timers:
        del camera_timers[camera_id]
    
    return jsonify({'success': True, 'message': 'Camera removed'})

@app.route('/api/cameras/upload', methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400
    
    file = request.files['video_file']
    name = request.form.get('name', 'Uploaded Video')
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({'success': False, 'message': f'Unsupported file type: {file_ext}'}), 400
    
    # Secure the filename
    filename = secure_filename(file.filename)
    timestamp = int(time.time() * 1000)
    filename = f"video_{timestamp}_{filename}"
    filepath = os.path.join(camera_manager.upload_folder, filename)
    
    try:
        file.save(filepath)
        file_size = os.path.getsize(filepath)
        logger.info(f"Video saved: {filepath} ({file_size / (1024*1024):.2f} MB)")
        
        # Verify video is readable
        test_cap = cv2.VideoCapture(filepath)
        if not test_cap.isOpened():
            test_cap.release()
            os.remove(filepath)
            return jsonify({'success': False, 'message': 'Invalid video file - cannot be read'}), 400
        
        ret, _ = test_cap.read()
        test_cap.release()
        
        if not ret:
            os.remove(filepath)
            return jsonify({'success': False, 'message': 'Video file is empty or corrupted'}), 400
        
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({'success': False, 'message': f'Failed to save file: {str(e)}'}), 500
    
    # Add as camera source
    camera_id = f"cam_{timestamp}"
    camera_manager.add_camera(camera_id, name, filepath)
    success = camera_manager.start_camera(camera_id)
    
    if success:
        return jsonify({
            'success': True, 
            'camera_id': camera_id, 
            'message': f'Video uploaded: {name}',
            'file_size_mb': f"{file_size / (1024*1024):.2f}"
        })
    else:
        camera_manager.remove_camera(camera_id)
        return jsonify({'success': False, 'message': 'Failed to start camera stream'}), 500

def generate_frames(camera_id):
    """Generator function for video streaming with fall detection"""
    fps_limit = 15  # Limit FPS for cloud deployment
    frame_delay = 1.0 / fps_limit
    frame_count = 0
    last_fps_time = time.time()
    current_fps = 0
    
    while True:
        start_time = time.time()
        frame = camera_manager.get_frame(camera_id)
        
        if frame is None:
            # Send a blank frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, 'No Signal', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(blank, camera_manager.cameras.get(camera_id, {}).get('name', 'Camera'), 
                       (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            ret, buffer = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
        else:
            # Initialize fall timer if needed
            if camera_id not in camera_timers:
                camera_timers[camera_id] = FallTimer(threshold=camera_manager.settings['fall_delay_seconds'])
            
            # Process frame for fall detection
            pose_result = pose_processor.process_frame(frame, camera_id)
            
            is_falling = False
            fall_confidence = 0.0
            
            if pose_result and pose_result.get('landmarks'):
                landmarks = pose_result['landmarks']
                
                # Try ML model detection first
                if LSTM_MODEL is not None and MODEL_AVAILABLE:
                    try:
                        # Extract features
                        if 'extract_55_features' in dir():
                            feature_vec = extract_55_features(frame, pose_processor.pose)
                        else:
                            feature_vec = extract_features_fallback(landmarks)
                        
                        # Add to sequence
                        camera_manager.pose_sequences[camera_id].append(feature_vec)
                        
                        # Predict if sequence is ready
                        if len(camera_manager.pose_sequences[camera_id]) >= SEQUENCE_LENGTH:
                            input_data = np.array(camera_manager.pose_sequences[camera_id], dtype=np.float32)
                            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                output = LSTM_MODEL(input_tensor)
                                
                                # Handle different output formats
                                if OUTPUT_SIZE == 1:
                                    prob = torch.sigmoid(output).item()
                                else:
                                    prob = torch.softmax(output, dim=1)[0][1].item()
                                
                                fall_confidence = prob
                                is_falling = prob >= camera_manager.settings['fall_threshold']
                    
                    except Exception as e:
                        logger.error(f"ML prediction error: {e}")
                        # Fallback to heuristic
                        is_falling, fall_confidence = detect_fall_heuristic(landmarks)
                else:
                    # Use heuristic detection
                    is_falling, fall_confidence = detect_fall_heuristic(landmarks)
            
            # Update fall timer
            fall_detected = camera_timers[camera_id].update(is_falling)
            
            # Update camera status
            camera_manager.cameras[camera_id]['confidence_score'] = fall_confidence
            
            if fall_detected:
                camera_manager.cameras[camera_id]['color'] = 'red'
                camera_manager.cameras[camera_id]['status'] = 'FALL DETECTED'
                logger.warning(f"⚠️ FALL DETECTED on {camera_manager.cameras[camera_id]['name']} (confidence: {fall_confidence:.2%})")
            elif is_falling:
                camera_manager.cameras[camera_id]['color'] = 'yellow'
                camera_manager.cameras[camera_id]['status'] = 'Analyzing'
            else:
                camera_manager.cameras[camera_id]['color'] = 'green'
                camera_manager.cameras[camera_id]['status'] = 'Normal'
            
            # Draw overlay on frame
            overlay_color = (0, 255, 0)  # Green
            if fall_detected:
                overlay_color = (0, 0, 255)  # Red
            elif is_falling:
                overlay_color = (0, 255, 255)  # Yellow
            
            # Draw pose landmarks if available
            if pose_result and pose_result.get('landmarks'):
                # Simple skeleton drawing
                h, w = frame.shape[:2]
                for lm in pose_result['landmarks'][:33]:
                    if lm['visibility'] > 0.5:
                        cx, cy = int(lm['x'] * w), int(lm['y'] * h)
                        cv2.circle(frame, (cx, cy), 3, overlay_color, -1)
            
            # Draw status box
            status_text = camera_manager.cameras[camera_id]['status']
            conf_text = f"Conf: {fall_confidence:.1%}"
            
            cv2.rectangle(frame, (10, 10), (350, 100), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 10), (350, 100), (255, 255, 255), 2)
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)
            cv2.putText(frame, conf_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Resize frame for lower bandwidth
            frame = cv2.resize(frame, (640, 480))
            
            # Encode frame with compression
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_bytes = buffer.tobytes()
            
            # Update FPS
            frame_count += 1
            if time.time() - last_fps_time >= 1.0:
                current_fps = frame_count / (time.time() - last_fps_time)
                camera_manager.cameras[camera_id]['fps'] = round(current_fps, 1)
                frame_count = 0
                last_fps_time = time.time()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Break if camera is stopped
        if camera_id not in camera_manager.camera_streams:
            break
        
        # Control frame rate
        elapsed = time.time() - start_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Video streaming route"""
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy', 
        'cameras': len(camera_manager.cameras),
        'model_loaded': LSTM_MODEL is not None,
        'active_streams': len(camera_manager.camera_streams)
    }), 200

@app.route('/api/debug/cameras', methods=['GET'])
def debug_cameras():
    """Debug endpoint to see camera states"""
    debug_info = {
        'cameras': list(camera_manager.cameras.values()),
        'active_streams': list(camera_manager.camera_streams.keys()),
        'settings': camera_manager.settings,
        'model_loaded': LSTM_MODEL is not None,
        'model_available': MODEL_AVAILABLE
    }
    return jsonify(debug_info)

# ----------------------------------------------------
# ERROR HANDLERS
# ----------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# ----------------------------------------------------
# STARTUP
# ----------------------------------------------------

def initialize_default_camera():
    """Initialize a default camera on startup"""
    DEFAULT_CAMERA_ID = "main_webcam_0"
    DEFAULT_CAMERA_NAME = "Main Webcam"
    DEFAULT_CAMERA_SOURCE = "0"  # Webcam 0
    
    logger.info(f"Initializing default camera: {DEFAULT_CAMERA_NAME}")
    camera_manager.add_camera(DEFAULT_CAMERA_ID, DEFAULT_CAMERA_NAME, DEFAULT_CAMERA_SOURCE)
    
    # Don't auto-start webcam on cloud (may not exist)
    # User will need to add cameras manually
    logger.info("Default camera added (not started). Add cameras via UI.")

# ----------------------------------------------------
# RUN
# ----------------------------------------------------

if __name__ == '__main__':
    print("\n" + "="*60)
    print("   FALLGUARD - AI Fall Detection System")
    print("="*60)
    print(f"\n[INFO] Model Status: {'✓ LSTM Loaded' if LSTM_MODEL else '✗ Heuristic Only'}")
    print(f"[INFO] MediaPipe: {'✓ Available' if mp_pose else '✗ Unavailable'}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Feature Size: {FEATURE_SIZE}")
    print(f"[INFO] Sequence Length: {SEQUENCE_LENGTH}")
    
    # Initialize default camera
    initialize_default_camera()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n[INFO] Starting server on port {port}")
    print(f"[INFO] Access at: http://localhost:{port}")
    print(f"[INFO] Health check: http://localhost:{port}/health")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)