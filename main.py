import os
import threading
import time
import json
import uuid
from flask import Flask, request, jsonify, session, send_from_directory, Response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename 
import numpy as np
import cv2
from collections import deque

# --- IMPORT MODULES ---
import torch
from app.skeleton_lstm import LSTMModel, SEQUENCE_LENGTH, FEATURE_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE
from app.video_utils import extract_55_features, draw_skeleton, predict_torch 
import mediapipe as mp

# --- Global Settings ---
DEFAULT_FALL_THRESHOLD = 0.70
INTERNAL_FPS = 30
DEFAULT_FALL_DELAY_SECONDS = 2
GLOBAL_SETTINGS = {
    "fall_threshold": DEFAULT_FALL_THRESHOLD,
    "fall_delay_seconds": DEFAULT_FALL_DELAY_SECONDS
}

# --- Model Loading ---
MODEL_FILE = 'models/skeleton_lstm_pytorch_model.pth'
LSTM_MODEL = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Initializing PyTorch. Using device: {device}")

try:
    # Try loading with the correct output size first
    try:
        LSTM_MODEL = LSTMModel(FEATURE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
        
        if os.path.exists(MODEL_FILE):
            LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
            LSTM_MODEL.to(device)
            LSTM_MODEL.eval()
            print(f"[SUCCESS] Loaded LSTM Model from {MODEL_FILE}")
            print(f"           Features: {FEATURE_SIZE}, Hidden: {HIDDEN_SIZE}, Output: {OUTPUT_SIZE}")
        else:
            print(f"[WARNING] Model file not found: {MODEL_FILE}")
            LSTM_MODEL = None
    except RuntimeError as e:
        # Model shape mismatch - try with output_size=1 (older model format)
        if "size mismatch" in str(e):
            print(f"[INFO] Model shape mismatch detected, trying legacy format (output_size=1)")
            LSTM_MODEL = LSTMModel(FEATURE_SIZE, HIDDEN_SIZE, 1, NUM_LAYERS)
            
            if os.path.exists(MODEL_FILE):
                LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
                LSTM_MODEL.to(device)
                LSTM_MODEL.eval()
                print(f"[SUCCESS] Loaded LSTM Model (legacy format) from {MODEL_FILE}")
            else:
                LSTM_MODEL = None
        else:
            raise
except Exception as e:
    print(f"[ERROR] Failed to load LSTM model: {e}") 
    print(f"[INFO] Falling back to enhanced heuristic detection")
    LSTM_MODEL = None

# MediaPipe Setup
USE_MEDIAPIPE = False
try:
    mp_pose = mp.solutions.pose
    USE_MEDIAPIPE = True
    print("[SUCCESS] MediaPipe initialized successfully")
except Exception as e:
    print(f"[ERROR] MediaPipe not available: {e}")

# Fall Timer Logic
class FallTimer:
    def __init__(self, threshold_frames=5):
        self.threshold = threshold_frames
        self.counter = 0
        self.last_fall_time = 0
    
    def update(self, is_falling):
        current_time = time.time()
        if is_falling:
            self.counter += 1
            self.last_fall_time = current_time
        else:
            if current_time - self.last_fall_time > 1.0:
                self.counter = 0
        return self.counter >= self.threshold

# --- GLOBAL CAMERA MANAGEMENT (FIXED) ---
CAMERA_DEFINITIONS = {}  # Persistent camera definitions
CAMERA_STATUS = {}       # Live status of cameras
shared_frames = {}       # Video frames for streaming
camera_lock = threading.Lock()

# Flask app 
app = Flask(__name__, static_folder='app', static_url_path='')
app.secret_key = os.environ.get("FALLGUARD_SECRET", "fallguard_secret_key_2024")
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# --- Enhanced Camera Processor (FIXED) ---
class CameraProcessor(threading.Thread):
    def __init__(self, camera_id, src, name, sequence_length=SEQUENCE_LENGTH, device=None):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.src = src 
        self.name = name
        self.cap = None
        self.is_running = False
        self.device = device if device is not None else torch.device('cpu')
        
        self.fall_timer = FallTimer(threshold_frames=1) 
        
        self.mp_pose_instance = None
        if USE_MEDIAPIPE:
            self.mp_pose_instance = mp_pose.Pose(
                static_image_mode=False, 
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                enable_segmentation=False,
                smooth_landmarks=True
            ) 
        
        self.sequence_length = sequence_length
        self.pose_sequence = deque([np.zeros(FEATURE_SIZE, dtype=np.float32) for _ in range(sequence_length)], 
                                     maxlen=sequence_length) 
        
        self.latest_pose_results = None
        self.latest_fall_prob = 0.0
        self.latest_features = None
        
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0
        self.processing_time = 0
        
        # FIXED: Initialize shared frame immediately
        self._init_shared_frame()

    def _init_shared_frame(self):
        """Initialize shared frame with placeholder."""
        placeholder = 100 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Initializing...", (180, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(placeholder, self.name, (200, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        
        shared_frames[self.camera_id] = {
            "frame": placeholder,
            "lock": threading.Lock()
        }

    def update_fall_timer_threshold(self):
        """Updates fall timer based on global settings."""
        delay_seconds = GLOBAL_SETTINGS['fall_delay_seconds']
        frame_threshold = max(1, round(delay_seconds * INTERNAL_FPS)) 
        self.fall_timer = FallTimer(threshold_frames=frame_threshold)

    def update_camera_status(self, status, color, last_alert=None, is_live=True):
        with camera_lock:
            if self.camera_id not in CAMERA_STATUS:
                CAMERA_STATUS[self.camera_id] = {}
            
            CAMERA_STATUS[self.camera_id].update({
                "status": status,
                "color": color,
                "isLive": is_live,
                "name": self.name,
                "source": str(self.src),
                "confidence_score": self.latest_fall_prob,
                "model_threshold": GLOBAL_SETTINGS['fall_threshold'],
                "fps": self.current_fps
            })
            if last_alert:
                CAMERA_STATUS[self.camera_id]["lastAlert"] = time.ctime(last_alert)

    def extract_features_and_bbox(self, frame):
        """Extract pose features using MediaPipe."""
        if USE_MEDIAPIPE and self.mp_pose_instance:
            try:
                bbox, feature_vec, pose_results = extract_55_features(frame, self.mp_pose_instance)
                self.latest_pose_results = pose_results
                self.latest_features = feature_vec
                
                # Debug: Check if we're getting valid features
                if feature_vec is not None and np.any(feature_vec != 0):
                    # Valid features detected
                    pass
                else:
                    # No person detected - use zero features
                    feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
                    bbox = None
                    pose_results = None
            except Exception as e:
                print(f"[ERROR] Feature extraction failed: {e}")
                feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
                bbox = None
                pose_results = None
        else:
            feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
            bbox = None
            pose_results = None
            
        self.pose_sequence.append(feature_vec)
        return bbox, feature_vec, pose_results

    def predict_fall_enhanced(self):
        """Enhanced fall prediction."""
        current_threshold = GLOBAL_SETTINGS['fall_threshold']
        fall_probability = 0.0

        # Always try heuristic detection first as a baseline
        if len(self.pose_sequence) > 0 and self.latest_features is not None:
            features = self.latest_features
            
            # Extract key features
            HWR = features[0]           # Height-Width Ratio
            TorsoAngle = features[1]    # Torso angle
            D = features[2]             # Head-Hip difference
            H = features[5]             # Hip height (normalized)
            FallAngleD = features[6]    # Fall angle
            
            # Enhanced heuristic scoring
            fall_score = 0.0
            
            # Criterion 1: Low HWR (person wider than tall)
            if 0.0 < HWR < 0.7:
                fall_score += 0.3
                if HWR < 0.5:
                    fall_score += 0.2
            
            # Criterion 2: High torso angle (leaning/horizontal)
            if TorsoAngle > 45:
                fall_score += 0.25
                if TorsoAngle > 70:
                    fall_score += 0.15
            
            # Criterion 3: Low hip height
            if H > 0.6:
                fall_score += 0.2
                if H > 0.75:
                    fall_score += 0.2
            
            # Criterion 4: Low fall angle (body horizontal)
            if FallAngleD < 30:
                fall_score += 0.3
            
            # Criterion 5: Small head-hip difference
            if abs(D) < 0.15:
                fall_score += 0.15
            
            fall_probability = min(fall_score, 0.99)
            
            # Debug output every 30 frames
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 0
                
            if self.debug_counter % 30 == 0:
                print(f"[{self.name}] Heuristic: HWR={HWR:.2f}, Torso={TorsoAngle:.0f}°, H={H:.2f}, Angle={FallAngleD:.0f}°, Score={fall_probability:.2f}")

        # Try LSTM model if available and sequence is ready
        if LSTM_MODEL is not None and len(self.pose_sequence) >= self.sequence_length:
            try:
                input_data = np.array(self.pose_sequence, dtype=np.float32)
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred, prob = predict_torch(LSTM_MODEL, input_tensor, threshold=current_threshold)
                
                # Use LSTM probability if it's higher than heuristic
                if prob > fall_probability:
                    fall_probability = prob
                    if self.debug_counter % 30 == 0:
                        print(f"[{self.name}] LSTM override: {prob:.2f}")
            except Exception as e:
                print(f"[ERROR] LSTM prediction failed for {self.camera_id}: {e}")
                # Continue with heuristic probability

        self.latest_fall_prob = fall_probability
        return (fall_probability >= current_threshold), fall_probability

    def draw_enhanced_overlay(self, frame, bbox, fall_confirmed, fall_prob, current_threshold, feature_vector):
        """Enhanced visualization."""
        h, w, _ = frame.shape
        
        panel_height = 130
        panel_color = (20, 20, 20) if not fall_confirmed else (0, 0, 80)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (350, panel_height), panel_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (5, 5), (350, panel_height), (255, 255, 255), 2)
        
        y_offset = 28
        line_height = 25
        
        status_text = "⚠️ FALL DETECTED!" if fall_confirmed else "✓ Monitoring Active"
        status_color = (0, 0, 255) if fall_confirmed else (0, 255, 0)
        cv2.putText(frame, status_text, (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
        
        y_offset += line_height
        cv2.putText(frame, f"Confidence: {fall_prob:.1%}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        y_offset += line_height
        cv2.putText(frame, f"Threshold: {current_threshold:.1%}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        y_offset += line_height
        cv2.putText(frame, f"Alert Delay: {GLOBAL_SETTINGS['fall_delay_seconds']}s", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        y_offset += line_height
        cv2.putText(frame, f"FPS: {self.current_fps:.1f} | Model: {'LSTM' if LSTM_MODEL else 'Heuristic'}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
            x, y, bw, bh = bbox
            color = (0, 0, 255) if fall_confirmed else (0, 255, 0)
            thickness = 5 if fall_confirmed else 2
            
            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), color, thickness)
            
            label = "FALL" if fall_confirmed else "PERSON"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            label_x = int(x + (bw - label_size[0]) / 2)
            label_y = max(30, int(y - 15))
            
            cv2.rectangle(frame, (label_x - 8, label_y - label_size[1] - 8), 
                         (label_x + label_size[0] + 8, label_y + 8), color, -1)
            cv2.putText(frame, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame

    def run(self):
        """Main processing loop - FIXED."""
        self.update_fall_timer_threshold()
        
        # Update status to starting
        self.update_camera_status("Starting...", "gray", is_live=True)
        
        print(f"[{self.name}] Opening video source: {self.src}")
        
        # Try multiple times to open camera
        max_retries = 3
        for attempt in range(max_retries):
            self.cap = cv2.VideoCapture(self.src)
            
            if self.cap and self.cap.isOpened():
                # Successfully opened
                ret, test_frame = self.cap.read()
                if ret:
                    print(f"[SUCCESS] Camera '{self.name}' opened on attempt {attempt + 1}")
                    break
                else:
                    self.cap.release()
                    self.cap = None
            
            if attempt < max_retries - 1:
                print(f"[RETRY] Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)

        if not self.cap or not self.cap.isOpened():
            print(f"[ERROR] Failed to open camera: {self.src}")
            self.update_camera_status("Failed to Open", "gray", is_live=False)
            
            # Create error frame
            error_frame = 100 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Error", (180, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.putText(error_frame, self.name, (200, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            
            with shared_frames[self.camera_id]["lock"]:
                shared_frames[self.camera_id]["frame"] = error_frame
            
            with camera_lock:
                if self.camera_id in CAMERA_DEFINITIONS:
                    CAMERA_DEFINITIONS[self.camera_id]['isLive'] = False
                    CAMERA_DEFINITIONS[self.camera_id]['thread_instance'] = None
            return

        # Determine if video file
        is_video_file = isinstance(self.src, str) and not str(self.src).isdigit() and os.path.exists(self.src)

        # Set camera properties
        if not is_video_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[{self.name}] Video file: FPS={video_fps}, Frames={total_frames}")

        # Get first frame
        ret, first_frame = self.cap.read()
        if ret:
            first_frame = cv2.resize(first_frame, (640, 480))
            with shared_frames[self.camera_id]["lock"]:
                shared_frames[self.camera_id]["frame"] = first_frame
            
            # Reset position for video files
            if is_video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.is_running = True
        self.update_camera_status("Active", "green", is_live=True)
        
        print(f"[SUCCESS] Camera '{self.name}' started successfully")

        # Main loop
        consecutive_failures = 0
        max_failures = 30  # Allow 30 consecutive failures before giving up
        
        try:
            while self.is_running:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                
                # Handle video looping
                if is_video_file and not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                    ret, frame = self.cap.read()

                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"[ERROR] {self.name}: Too many consecutive failures")
                        break
                    time.sleep(0.05)
                    continue
                
                consecutive_failures = 0  # Reset on success
                
                # Process frame
                frame = cv2.resize(frame, (640, 480))
                
                try:
                    bbox, feature_vec, pose_results = self.extract_features_and_bbox(frame)
                except Exception as e:
                    bbox, feature_vec, pose_results = None, None, None

                is_falling, fall_prob = self.predict_fall_enhanced()
                fall_confirmed = self.fall_timer.update(is_falling) 
                current_threshold = GLOBAL_SETTINGS['fall_threshold']

                if fall_confirmed:
                    self.update_camera_status("FALL DETECTED", "red", last_alert=time.time())
                elif is_falling:
                    self.update_camera_status("Analyzing", "yellow")
                else:
                    self.update_camera_status("Normal", "green")

                # Draw visualizations
                processed = frame.copy()
                
                if self.latest_pose_results:
                    processed = draw_skeleton(processed, self.latest_pose_results, fall_confirmed)

                processed = self.draw_enhanced_overlay(processed, bbox, fall_confirmed, 
                                                      fall_prob, current_threshold, feature_vec)

                # Store frame
                with shared_frames[self.camera_id]["lock"]:
                    shared_frames[self.camera_id]["frame"] = processed

                # FPS calculation
                self.frame_count += 1
                if time.time() - self.last_fps_update >= 1.0:
                    self.current_fps = self.frame_count / (time.time() - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = time.time()

                # Frame rate control
                processing_time = time.time() - start_time
                if is_video_file:
                    fps = self.cap.get(cv2.CAP_PROP_FPS) or INTERNAL_FPS 
                    target_delay = 1.0 / fps
                    sleep_time = max(0, target_delay - processing_time)
                    time.sleep(sleep_time)
                else:
                    target_delay = 1.0 / INTERNAL_FPS
                    sleep_time = max(0, target_delay - processing_time)
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"[ERROR] Camera processor crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if self.mp_pose_instance: 
                self.mp_pose_instance.close()
            if self.cap: 
                self.cap.release()
            
            with camera_lock:
                if self.camera_id in CAMERA_STATUS: 
                    del CAMERA_STATUS[self.camera_id]
                if self.camera_id in CAMERA_DEFINITIONS: 
                    CAMERA_DEFINITIONS[self.camera_id]['isLive'] = False
                    CAMERA_DEFINITIONS[self.camera_id]['thread_instance'] = None
                
            print(f"[INFO] Camera '{self.name}' stopped")

# MJPEG Stream Generator (FIXED)
def generate_mjpeg(camera_id):
    """Generates MJPEG stream for a camera."""
    boundary = b'--frame\r\n'
    
    # Wait briefly for camera to initialize
    wait_time = 0
    max_wait = 5  # Wait up to 5 seconds
    
    while camera_id not in shared_frames and wait_time < max_wait:
        time.sleep(0.1)
        wait_time += 0.1
    
    if camera_id not in shared_frames:
        placeholder = 100 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera Not Available", (150, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(placeholder, f"ID: {camera_id}", (200, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        
        ret, jpeg = cv2.imencode('.jpg', placeholder)
        frame_bytes = jpeg.tobytes()
        yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        return
        
    while camera_id in shared_frames:
        frame_data = shared_frames[camera_id]
        with frame_data["lock"]:
            frame = frame_data["frame"].copy() if frame_data["frame"] is not None else None
        
        if frame is None:
            placeholder = 100 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Initializing...", (180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', placeholder)
            frame_bytes = jpeg.tobytes()
        else:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = jpeg.tobytes()

        yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        time.sleep(0.033)

# Flask Routes
@app.route('/')
def index():
    return send_from_directory('app', 'index.html')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    return Response(generate_mjpeg(camera_id), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'POST':
        data = request.get_json() or {}
        message = []
        
        new_threshold = data.get('fall_threshold')
        if new_threshold is not None:
            try:
                new_threshold = float(new_threshold)
                if 0.0 <= new_threshold <= 1.0:
                    GLOBAL_SETTINGS['fall_threshold'] = new_threshold
                    message.append("Threshold updated")
                else:
                    return jsonify({"success": False, "message": "Threshold must be 0.0-1.0"}), 400
            except ValueError:
                return jsonify({"success": False, "message": "Invalid threshold value"}), 400
                
        new_delay = data.get('fall_delay_seconds')
        if new_delay is not None:
            try:
                new_delay = int(new_delay)
                if 1 <= new_delay <= 10: 
                    GLOBAL_SETTINGS['fall_delay_seconds'] = new_delay
                    message.append("Delay updated")
                    
                    with camera_lock:
                        for cam_def in CAMERA_DEFINITIONS.values():
                            processor = cam_def.get('thread_instance')
                            if processor and processor.is_running:
                                processor.update_fall_timer_threshold()
                else:
                    return jsonify({"success": False, "message": "Delay must be 1-10 seconds"}), 400
            except ValueError:
                return jsonify({"success": False, "message": "Invalid delay value"}), 400

        return jsonify({"success": True, "message": " ".join(message), "settings": GLOBAL_SETTINGS})
        
    return jsonify({"success": True, "settings": GLOBAL_SETTINGS})

@app.route('/api/cameras', methods=['GET'])
def api_get_cameras():
    cameras = []
    with camera_lock:
        for cam_id, cam_def in CAMERA_DEFINITIONS.items():
            processor = cam_def.get('thread_instance')
            
            is_actually_live = False
            if processor is not None:
                try:
                    is_actually_live = processor.is_running and processor.is_alive()
                except:
                    is_actually_live = False
            
            status = CAMERA_STATUS.get(cam_id, {
                "status": "Offline", 
                "color": "gray", 
                "isLive": False, 
                "confidence_score": 0.0,
                "fps": 0
            })
            
            actual_is_live = is_actually_live and cam_id in shared_frames
            
            cameras.append({
                "id": cam_id,
                "name": cam_def['name'],
                "source": str(cam_def['source']),
                "isLive": actual_is_live,
                "status": status['status'] if actual_is_live else "Offline",
                "color": status['color'] if actual_is_live else "gray",
                "lastAlert": status.get('lastAlert', 'N/A'),
                "confidence_score": status.get('confidence_score', 0.0),
                "model_threshold": GLOBAL_SETTINGS['fall_threshold'],
                "fps": status.get('fps', 0)
            })
    
    return jsonify({"success": True, "cameras": cameras})

@app.route('/api/cameras/all_definitions', methods=['GET'])
def api_get_all_definitions():
    definitions = []
    with camera_lock:
        for cam_id, cam_def in CAMERA_DEFINITIONS.items():
            processor = cam_def.get('thread_instance')
            is_live = False
            if processor is not None:
                try:
                    is_live = processor.is_running and processor.is_alive()
                except:
                    is_live = False
            
            definitions.append({
                "id": cam_id,
                "name": cam_def['name'],
                "source": str(cam_def['source']),
                "isLive": is_live
            })
    return jsonify({"success": True, "definitions": definitions})

@app.route('/api/cameras/add', methods=['POST'])
def api_add_camera():
    data = request.get_json()
    name = data.get('name')
    source_str = data.get('source')
    
    if not name or source_str is None:
        return jsonify({"success": False, "message": "Name and source required"}), 400

    try:
        source = int(source_str)
    except ValueError:
        source = source_str
    
    camera_id = f"cam_{str(uuid.uuid4())[:8]}"
    
    processor = CameraProcessor(camera_id=camera_id, src=source, name=name, device=device)
    processor.start()
    
    with camera_lock:
        CAMERA_DEFINITIONS[camera_id] = {
            "name": name, 
            "source": source, 
            "isLive": True,
            "thread_instance": processor
        }

    return jsonify({"success": True, "message": f"Camera '{name}' added", "camera_id": camera_id})

@app.route('/api/cameras/stop/<camera_id>', methods=['POST'])
def api_stop_camera(camera_id):
    with camera_lock:
        if camera_id not in CAMERA_DEFINITIONS:
            return jsonify({"success": False, "message": "Camera not found"}), 404

        cam_def = CAMERA_DEFINITIONS[camera_id]
        processor = cam_def.get('thread_instance')

        if processor and processor.is_running:
            processor.is_running = False
            processor.join(timeout=3)

        CAMERA_DEFINITIONS[camera_id]['isLive'] = False
        CAMERA_DEFINITIONS[camera_id]['thread_instance'] = None
        
        if camera_id in CAMERA_STATUS:
            del CAMERA_STATUS[camera_id]

    return jsonify({"success": True, "message": "Camera stopped"})

@app.route('/api/cameras/remove/<camera_id>', methods=['DELETE'])
def api_remove_camera(camera_id):
    with camera_lock:
        if camera_id not in CAMERA_DEFINITIONS:
            return jsonify({"success": False, "message": "Camera not found"}), 404

        cam_def = CAMERA_DEFINITIONS[camera_id]
        processor = cam_def.get('thread_instance')
        if processor and processor.is_running:
            processor.is_running = False
            processor.join(timeout=3)

        # Delete video file if it exists
        source = cam_def['source']
        if isinstance(source, str) and source.startswith(UPLOAD_FOLDER):
            try:
                if os.path.exists(source):
                    os.remove(source)
                    print(f"[INFO] Deleted video file: {source}")
            except Exception as e:
                print(f"[WARNING] Could not delete file {source}: {e}")

        del CAMERA_DEFINITIONS[camera_id]
        if camera_id in CAMERA_STATUS:
            del CAMERA_STATUS[camera_id]
        if camera_id in shared_frames:
            del shared_frames[camera_id]

    return jsonify({"success": True, "message": "Camera removed"})

@app.route('/api/cameras/add_existing', methods=['POST'])
def api_add_existing_camera():
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if not camera_id:
        return jsonify({"success": False, "message": "Camera ID required"}), 400

    with camera_lock:
        if camera_id not in CAMERA_DEFINITIONS:
            return jsonify({"success": False, "message": "Camera not found"}), 404
        
        cam_def = CAMERA_DEFINITIONS[camera_id]
        
        if cam_def.get('thread_instance') and cam_def['thread_instance'].is_running:
            return jsonify({"success": False, "message": "Camera already running"}), 400
        
        src_type = cam_def['source']
        try:
            if isinstance(src_type, str) and src_type.isdigit():
                src_type = int(src_type)
        except:
            pass

        processor = CameraProcessor(camera_id=camera_id, src=src_type, name=cam_def['name'], device=device)
        processor.start()
        
        CAMERA_DEFINITIONS[camera_id]['isLive'] = True
        CAMERA_DEFINITIONS[camera_id]['thread_instance'] = processor
        
    return jsonify({"success": True, "message": "Camera restarted"})

@app.route('/api/cameras/upload', methods=['POST'])
def api_upload_camera():
    if 'video_file' not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400
    
    file = request.files['video_file']
    name = request.form.get('name')

    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"}), 400
    if not name:
        return jsonify({"success": False, "message": "Camera name required"}), 400
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({"success": False, "message": f"Unsupported file type: {file_ext}"}), 400
    
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        print(f"[UPLOAD] Saving file: {filename} -> {filepath}")
        file.save(filepath)
        print(f"[UPLOAD] File saved successfully: {filepath}")
        
        # Verify file exists and is readable
        if not os.path.exists(filepath):
            return jsonify({"success": False, "message": "File save failed"}), 500
            
        file_size = os.path.getsize(filepath)
        print(f"[UPLOAD] File size: {file_size / (1024*1024):.2f} MB")
        
        # Quick validation that OpenCV can read it
        test_cap = cv2.VideoCapture(filepath)
        if not test_cap.isOpened():
            test_cap.release()
            os.remove(filepath)
            return jsonify({"success": False, "message": "Invalid video file - cannot be read"}), 400
        
        ret, _ = test_cap.read()
        test_cap.release()
        
        if not ret:
            os.remove(filepath)
            return jsonify({"success": False, "message": "Video file is empty or corrupted"}), 400
        
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({"success": False, "message": f"Upload failed: {str(e)}"}), 500

    camera_id = f"cam_{str(uuid.uuid4())[:8]}"
    name_safe = name

    print(f"[UPLOAD] Starting camera processor for: {name_safe} (ID: {camera_id})")
    
    processor = CameraProcessor(camera_id=camera_id, src=filepath, name=name_safe, device=device)
    processor.start()
    
    with camera_lock:
        CAMERA_DEFINITIONS[camera_id] = {
            "name": name_safe, 
            "source": filepath, 
            "isLive": True,
            "thread_instance": processor
        }
    
    print(f"[UPLOAD] Camera started: {name_safe}")
    
    return jsonify({
        "success": True, 
        "message": f"Video uploaded: {name_safe}", 
        "camera_id": camera_id,
        "file_size_mb": f"{file_size / (1024*1024):.2f}"
    })

@app.route('/api/debug/cameras', methods=['GET'])
def api_debug_cameras():
    """Debug endpoint to see camera states."""
    debug_info = {
        "definitions": {},
        "status": {},
        "shared_frames": list(shared_frames.keys()),
        "settings": GLOBAL_SETTINGS
    }
    
    with camera_lock:
        for cam_id, cam_def in CAMERA_DEFINITIONS.items():
            processor = cam_def.get('thread_instance')
            debug_info["definitions"][cam_id] = {
                "name": cam_def['name'],
                "source": str(cam_def['source']),
                "has_processor": processor is not None,
                "is_running": processor.is_running if processor else False,
                "is_alive": processor.is_alive() if processor else False,
                "in_shared_frames": cam_id in shared_frames
            }
        
        for cam_id, status in CAMERA_STATUS.items():
            debug_info["status"][cam_id] = status
    
    return jsonify(debug_info)

# Startup
if __name__ == '__main__':
    print("\n" + "="*60)
    print("   FALLGUARD - AI Fall Detection System")
    print("="*60)
    
    DEFAULT_CAMERA_ID = "main_webcam_0"
    DEFAULT_CAMERA_NAME = "Main Webcam"
    DEFAULT_CAMERA_SOURCE = 0

    print(f"\n[STARTUP] Initializing default camera: {DEFAULT_CAMERA_NAME}")
    print(f"[INFO] Source: {DEFAULT_CAMERA_SOURCE}")
    print(f"[INFO] Model: {'LSTM' if LSTM_MODEL else 'Heuristic-based'}")
    print(f"[INFO] MediaPipe: {'Enabled' if USE_MEDIAPIPE else 'Disabled'}")
    
    default_processor = CameraProcessor(
        camera_id=DEFAULT_CAMERA_ID, 
        src=DEFAULT_CAMERA_SOURCE, 
        name=DEFAULT_CAMERA_NAME,
        device=device
    )
    default_processor.start()

    with camera_lock:
        CAMERA_DEFINITIONS[DEFAULT_CAMERA_ID] = {
            "name": DEFAULT_CAMERA_NAME,
            "source": DEFAULT_CAMERA_SOURCE,
            "isLive": True,
            "thread_instance": default_processor
        }

    print(f"\n[INFO] Server starting on http://0.0.0.0:5000")
    print(f"[INFO] Access the system at: http://localhost:5000")
    print(f"[INFO] Debug endpoint: http://localhost:5000/api/debug/cameras")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)