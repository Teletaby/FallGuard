# main.py
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
# NOTE: These modules (skeleton_lstm, video_utils) are assumed to exist in 'app/' 
# and contain the necessary definitions (LSTMModel, extract_55_features, etc.)
import torch
from app.skeleton_lstm import LSTMModel, SEQUENCE_LENGTH, FEATURE_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE
from app.video_utils import extract_55_features, draw_skeleton, predict_torch 
import mediapipe as mp

# --- Global Settings ---
DEFAULT_FALL_THRESHOLD = 0.95 
INTERNAL_FPS = 30 # Assumption: Processor aims for 30 frames per second loop rate
DEFAULT_FALL_DELAY_SECONDS = 3 
GLOBAL_SETTINGS = {
    "fall_threshold": DEFAULT_FALL_THRESHOLD,
    "fall_delay_seconds": DEFAULT_FALL_DELAY_SECONDS
}

# --- Model Loading ---
MODEL_FILE = 'lstm_fall_model.pth' 
LSTM_MODEL = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Define device early
print(f"[INFO] Initializing PyTorch. Using device: {device}")

try:
    # 1. Initialize the model structure
    LSTM_MODEL = LSTMModel(FEATURE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    
    if os.path.exists(MODEL_FILE):
        # 2. Load the weights if the file exists
        LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        LSTM_MODEL.to(device) # Move model to device
        LSTM_MODEL.eval()
        print(f"[SUCCESS] Loaded PyTorch LSTM Model. Feat: {FEATURE_SIZE}, Seq: {SEQUENCE_LENGTH} on {device}")
    else:
        # Fallback if the model file is missing
        print(f"[ ERROR] LSTM Model file not found at {MODEL_FILE}. Using mock logic.") 
        LSTM_MODEL = None 
except Exception as e:
    # Handle fatal loading errors
    print(f"[FATAL] Failed to load PyTorch LSTM model: {e}") 
    LSTM_MODEL = None

# MediaPipe Setup Check
USE_MEDIAPIPE = False
try:
    mp_pose = mp.solutions.pose
    USE_MEDIAPIPE = True
    print("MediaPipe available: using pose detection.")
except Exception:
    print("MediaPipe not available. Fall detection will not work without it.")
    # Fallback setup 
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Fall Timer Logic
class FallTimer:
    """Manages the continuous frame count required to confirm a fall alert."""
    def __init__(self, threshold_frames=5):
        self.threshold = threshold_frames # Number of continuous frames/steps needed
        self.counter = 0
    def update(self, is_falling):
        if is_falling:
            self.counter += 1
        else:
            self.counter = 0
        return self.counter >= self.threshold

# --- GLOBAL CAMERA & STATUS MANAGEMENT ---
CAMERA_DEFINITIONS = {} 
CAMERA_STATUS = {} 
shared_frames = {}
camera_lock = threading.Lock() # Lock for modifying CAMERA_DEFINITIONS and CAMERA_STATUS

# -------------------------
# Flask app 
# -------------------------
# *** CRITICAL CHANGE: Set static_folder to 'app' ***
app = Flask(__name__, static_folder='app', static_url_path='')
app.secret_key = os.environ.get("FALLGUARD_SECRET", "super_secret_key")
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory users (Simple Authentication)
USERS = {}
USERS["guest@fallguard.com"] = {
    "id": "guest-id",
    "email": "guest@fallguard.com",
    "password": generate_password_hash("guest")
}


# -------------------------
# Camera Processing Thread
# -------------------------
class CameraProcessor(threading.Thread):
    def __init__(self, camera_id, src, name, sequence_length=SEQUENCE_LENGTH, device=None):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.src = src 
        self.name = name
        self.cap = None
        self.is_running = False
        self.device = device if device is not None else torch.device('cpu')
        
        # Initialize fall_timer placeholder
        self.fall_timer = FallTimer(threshold_frames=1) 
        
        self.mp_pose_instance = None
        if USE_MEDIAPIPE:
            # Separate MediaPipe instance for each thread
            self.mp_pose_instance = mp_pose.Pose(
                static_image_mode=False, 
                model_complexity=0, 
                enable_segmentation=False
            ) 
        
        self.sequence_length = sequence_length
        # Initialize deque with zero vectors matching FEATURE_SIZE (55)
        self.pose_sequence = deque([np.zeros(FEATURE_SIZE, dtype=np.float32) for _ in range(sequence_length)], 
                                     maxlen=sequence_length) 
        
        self.latest_pose_results = None
        self.latest_fall_prob = 0.0 # Store latest fall probability
        self.hog = None
        if not USE_MEDIAPIPE:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Dynamic update for the fall timer threshold
    def update_fall_timer_threshold(self):
        """Calculates the frame count needed based on global seconds delay."""
        delay_seconds = GLOBAL_SETTINGS['fall_delay_seconds']
        # Convert seconds to frame count threshold, ensuring it's at least 1
        frame_threshold = max(1, round(delay_seconds * INTERNAL_FPS)) 
        self.fall_timer = FallTimer(threshold_frames=frame_threshold)
        print(f"[{self.name}] Fall delay set to {delay_seconds}s (Threshold: {frame_threshold} frames).")

    def update_camera_status(self, status, color, last_alert=None, is_live=True):
        with camera_lock:
            if self.camera_id in CAMERA_STATUS:
                CAMERA_STATUS[self.camera_id].update({
                    "status": status,
                    "color": color,
                    "isLive": is_live,
                    "name": self.name,
                    "source": self.src,
                    "confidence_score": self.latest_fall_prob, # Add confidence score
                    "model_threshold": GLOBAL_SETTINGS['fall_threshold'] # Add model threshold
                })
                if last_alert:
                    CAMERA_STATUS[self.camera_id]["lastAlert"] = time.ctime(last_alert)

    def extract_features_and_bbox(self, frame):
        """Extracts 55 features using video_utils."""
        if USE_MEDIAPIPE and self.mp_pose_instance:
            # Use the integrated feature extraction from video_utils
            bbox, feature_vec, pose_results = extract_55_features(frame, self.mp_pose_instance)
            self.latest_pose_results = pose_results
        else:
            # Fallback (No useful features, feature_vec is zeros)
            feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
            bbox = None
            pose_results = None

            if self.hog:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Convert the frame to a format suitable for HOG detection
                (rects, _) = self.hog.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)
                if len(rects) > 0:
                    # Find the largest bounding box
                    bbox = max(rects, key=lambda r: r[2]*r[3])
            
        self.pose_sequence.append(feature_vec) # Append the new 55-dim feature vector
        return bbox, feature_vec, pose_results

    def predict_fall_lstm(self):
        current_threshold = GLOBAL_SETTINGS['fall_threshold']
        fall_probability = 0.0

        if LSTM_MODEL is None or len(self.pose_sequence) < self.sequence_length:
            # Mock logic when model is not loaded (or sequence not full)
            if LSTM_MODEL is None:
                # Use H (Normalized height of Hip Center) at index 5 for mock logic, if available
                h_com = self.pose_sequence[-1][5] 
                # Simple rule: if height is low and not zero (meaning person is detected)
                fall_probability = 0.95 if h_com < 0.2 and h_com > 0.0 else 0.05
            else:
                # Sequence not full but model loaded - use default low probability
                fall_probability = 0.01 
                
            self.latest_fall_prob = fall_probability # Store probability
            return (fall_probability >= current_threshold), fall_probability

        # Convert deque to a numpy array, then to a PyTorch tensor
        input_data = np.array(self.pose_sequence, dtype=np.float32)
        # Add batch dimension (1, seq_len, feature_size)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        try:
            # Use the integrated prediction function from video_utils
            pred, prob = predict_torch(LSTM_MODEL, input_tensor, threshold=current_threshold)
            self.latest_fall_prob = prob # Store probability
            return (pred == 1), prob
        except Exception as e:
            print(f"Prediction error for {self.camera_id}: {e}")
            self.latest_fall_prob = 0.0 # Store probability
            return False, 0.0

    def draw_bbox_and_status(self, frame, bbox, fall_confirmed, fall_prob, current_threshold, feature_vector):
        h, w, _ = frame.shape
        
        cv2.putText(frame, f"Prob: {fall_prob:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Threshold: {current_threshold:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Delay: {GLOBAL_SETTINGS['fall_delay_seconds']}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if bbox is not None and bbox[2] > 0 and bbox[3] > 0: # Ensure bbox is valid
            # Bbox from feature extraction is typically (x, y, w, h)
            x, y, bw, bh = bbox
            color = (0, 0, 255) if fall_confirmed else (0, 255, 0)
            thickness = 4 if fall_confirmed else 3
            
            # Draw Bounding Box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), color, thickness)
            
            label = "FALL DETECTED!" if fall_confirmed else "PERSON"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            # Ensure text is above the frame edge
            text_x = int(x + (bw - text_size[0]) / 2)
            text_y = max(15, int(y - 10))

            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Display HWR (index 0) and Fall Angle D (index 6) for debugging
            if feature_vector is not None and len(feature_vector) >= 7 and feature_vector[0] != 0:
                hwr = feature_vector[0] 
                fall_angle_D = feature_vector[6] 
                cv2.putText(frame, f"HWR: {hwr:.2f} | Angle(D): {fall_angle_D:.2f}", 
                            (int(x), int(y + bh + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return frame

    def run(self):
        self.update_fall_timer_threshold() # Set initial delay threshold before loop starts
        
        with camera_lock:
            CAMERA_STATUS[self.camera_id] = {
                "name": self.name, 
                "status": "Initializing", 
                "isLive": True, 
                "color": "gray", 
                "location": f"Source {self.src}",
                "source": self.src,
                "confidence_score": 0.0, # Initialize
                "model_threshold": GLOBAL_SETTINGS['fall_threshold'] # Initialize
            }
        
        # Open video source (Webcam index or file path)
        self.cap = cv2.VideoCapture(self.src)
        # Check if the source is a string (URL or file) but not a digit (webcam index)
        is_video_file = isinstance(self.src, str) and (not str(self.src).isdigit())

        if not self.cap or not self.cap.isOpened():
            print(f"ERROR: Cannot open video source {self.src}. Stopping thread for {self.camera_id}.")
            self.update_camera_status("Offline", "gray", is_live=False)
            
            # Clean up global state if camera failed to open
            with camera_lock:
                if self.camera_id in CAMERA_STATUS: del CAMERA_STATUS[self.camera_id]
                if self.camera_id in CAMERA_DEFINITIONS: 
                    CAMERA_DEFINITIONS[self.camera_id]['isLive'] = False
                    CAMERA_DEFINITIONS[self.camera_id]['thread_instance'] = None
            return

        shared_frames[self.camera_id] = {"frame": None, "lock": threading.Lock()}
        
        self.is_running = True
        self.update_camera_status("Running", "green", is_live=True)

        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if is_video_file and not ret:
                    # Loop video file if end is reached
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                    ret, frame = self.cap.read()

                if not ret:
                    time.sleep(0.05)
                    continue
                
                bbox = None
                feature_vec = None
                
                try:
                    # Resize frame for faster processing (optional, but good practice)
                    frame = cv2.resize(frame, (640, 480))
                    
                    bbox, feature_vec, pose_results = self.extract_features_and_bbox(frame)
                except Exception as e:
                    print(f"Feature extraction error for {self.camera_id}: {e}")
                    pose_results = None

                is_falling, fall_prob = self.predict_fall_lstm()
                # Update fall confirmation status using the FallTimer
                fall_confirmed = self.fall_timer.update(is_falling) 
                current_threshold = GLOBAL_SETTINGS['fall_threshold']

                # Update global status for the dashboard
                if fall_confirmed:
                    self.update_camera_status(f"FALL DETECTED ({fall_prob:.2f})", "red", last_alert=time.time())
                elif is_falling:
                    self.update_camera_status(f"Possible Fall ({fall_prob:.2f})", "yellow")
                else:
                    self.update_camera_status(f"Normal ({fall_prob:.2f})", "green")

                processed = frame.copy()
                
                # Draw skeleton if MediaPipe results are available
                if self.latest_pose_results:
                    processed = draw_skeleton(processed, self.latest_pose_results, fall_confirmed)

                # Draw BBox and status text
                processed = self.draw_bbox_and_status(processed, bbox, fall_confirmed, fall_prob, current_threshold, feature_vec)

                # Store the processed frame for MJPEG streaming
                with shared_frames[self.camera_id]["lock"]:
                    shared_frames[self.camera_id]["frame"] = processed

                # Control frame rate
                if is_video_file:
                    # Read FPS from file, default to INTERNAL_FPS
                    fps = self.cap.get(cv2.CAP_PROP_FPS) or INTERNAL_FPS 
                    if fps > 0:
                        delay = 1 / fps
                        time.sleep(delay)
                    else:
                        time.sleep(1/INTERNAL_FPS) # Fallback 
                else:
                    time.sleep(1/INTERNAL_FPS) # Maintain internal processing rate (30 FPS)

        finally:
            # Resource cleanup
            if self.mp_pose_instance: self.mp_pose_instance.close()
            if self.cap: self.cap.release()
            
            # Clean up global state when thread stops
            with camera_lock:
                if self.camera_id in CAMERA_STATUS: del CAMERA_STATUS[self.camera_id]
                if self.camera_id in shared_frames: del shared_frames[self.camera_id]

                if self.camera_id in CAMERA_DEFINITIONS: 
                    CAMERA_DEFINITIONS[self.camera_id]['isLive'] = False
                    CAMERA_DEFINITIONS[self.camera_id]['thread_instance'] = None
                
            print(f"Camera processor {self.camera_id} stopped.")

# -------------------------
# MJPEG stream generator
# -------------------------
def generate_mjpeg(camera_id):
    """Generator that yields multipart MJPEG frames for a specific camera_id."""
    # The boundary string used in the response header
    boundary = b'--frame\r\n'
    
    # Check if the camera is defined to be running
    if camera_id not in shared_frames:
        # Create a temporary black/grey image for when a stream is not found
        placeholder = 255 * np.ones((480, 640, 3), dtype=np.uint8) # Increased size slightly
        cv2.putText(placeholder, f"STREAM NOT FOUND: {camera_id}", (30,240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
        
        # Encode the placeholder image
        ret, jpeg = cv2.imencode('.jpg', placeholder)
        frame_bytes = jpeg.tobytes()
        
        # FIX: Explicitly concatenate bytes on a single line to resolve SyntaxError
        yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        return
        
    while camera_id in shared_frames:
        frame_data = shared_frames[camera_id]
        with frame_data["lock"]:
            frame = frame_data["frame"].copy() if frame_data["frame"] is not None else None
        
        if frame is None:
            # Placeholder for waiting or offline cameras
            placeholder = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Waiting for {camera_id}", (30,240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
            ret, jpeg = cv2.imencode('.jpg', placeholder)
            frame_bytes = jpeg.tobytes()
        else:
            # Use lower quality JPEG encoding for faster streaming
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = jpeg.tobytes()

        # FIX: Explicitly concatenate bytes on a single line to resolve SyntaxError
        yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        time.sleep(0.03) # Limit to roughly 33 FPS for streaming
        
    print(f"MJPEG generator stopped for {camera_id}.")


# -------------------------
# Flask Routes
# -------------------------
@app.route('/')
def index():
    # *** CRITICAL CHANGE: Look for index.html in the 'app' directory ***
    return send_from_directory('app', 'index.html')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    if camera_id not in CAMERA_STATUS and camera_id not in CAMERA_DEFINITIONS:
        # Return 404 if the camera doesn't exist at all
        return "Camera stream is not defined.", 404
        
    # generate_mjpeg will handle the case where it exists but isn't running
    return Response(generate_mjpeg(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- AUTH ROUTES ---
@app.route('/api/auth/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required"}), 400
    
    if email in USERS:
        return jsonify({"success": False, "message": "User already exists"}), 409
    
    user_id = str(uuid.uuid4())
    USERS[email] = {
        "id": user_id,
        "email": email,
        "password": generate_password_hash(password)
    }
    session['user_id'] = user_id
    session['email'] = email
    return jsonify({"success": True, "message": "User created and signed in.", "email": email, "userId": user_id})

@app.route('/api/auth/signin', methods=['POST'])
def api_signin():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required"}), 400

    user_data = USERS.get(email)
    if user_data and check_password_hash(user_data['password'], password):
        session['user_id'] = user_data['id']
        session['email'] = email
        return jsonify({"success": True, "message": "Signed in successfully.", "email": email, "userId": user_data['id']})
    
    return jsonify({"success": False, "message": "Invalid email or password"}), 401

@app.route('/api/auth/signout', methods=['POST'])
def api_signout():
    session.pop('user_id', None)
    session.pop('email', None)
    return jsonify({"success": True, "message": "Signed out successfully."})

@app.route('/api/status', methods=['GET'])
def api_status():
    if 'user_id' in session:
        return jsonify({
            "isLoggedIn": True, 
            "email": session.get('email'),
            "userId": session.get('user_id')
        })
    return jsonify({"isLoggedIn": False, "email": None, "userId": None})


# --- GLOBAL SETTINGS API (UPDATED) ---
@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if 'user_id' not in session: return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    if request.method == 'POST':
        data = request.get_json() or {}
        message = []
        
        # 1. Update Fall Probability Threshold
        new_threshold = data.get('fall_threshold')
        if new_threshold is not None:
            try:
                new_threshold = float(new_threshold)
                if 0.0 <= new_threshold <= 1.0:
                    GLOBAL_SETTINGS['fall_threshold'] = new_threshold
                    message.append("Threshold updated.")
                else:
                    return jsonify({"success": False, "message": "Threshold must be between 0.0 and 1.0"}), 400
            except ValueError:
                return jsonify({"success": False, "message": "Invalid fall_threshold value."}), 400
                
        # 2. Update Fall Delay Seconds
        new_delay_seconds = data.get('fall_delay_seconds')
        if new_delay_seconds is not None:
            try:
                # Expecting an integer number of seconds (e.g., 1 to 10)
                new_delay_seconds = int(new_delay_seconds)
                if 1 <= new_delay_seconds <= 10: 
                    GLOBAL_SETTINGS['fall_delay_seconds'] = new_delay_seconds
                    message.append("Delay updated.")
                    
                    # Apply new delay to all running camera processors
                    with camera_lock:
                        for cam_def in CAMERA_DEFINITIONS.values():
                            processor = cam_def.get('thread_instance')
                            if processor and processor.is_running:
                                # This ensures the FallTimer inside the thread is re-initialized
                                processor.update_fall_timer_threshold() 
                else:
                    return jsonify({"success": False, "message": "Delay must be between 1 and 10 seconds"}), 400
            except ValueError:
                return jsonify({"success": False, "message": "Invalid fall_delay_seconds value."}), 400

        return jsonify({"success": True, "message": " ".join(message) or "Settings checked, no changes applied.", "settings": GLOBAL_SETTINGS})
        
    return jsonify({"success": True, "settings": GLOBAL_SETTINGS})


# --- CAMERA MANAGEMENT API ---
@app.route('/api/cameras', methods=['GET'])
def api_get_cameras():
    if 'user_id' not in session: return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    # Merge definitions (permanent config) and status (live update)
    cameras = []
    with camera_lock:
        for cam_id, cam_def in CAMERA_DEFINITIONS.items():
            status = CAMERA_STATUS.get(cam_id, {
                "status": "Offline", 
                "color": "gray", 
                "isLive": False, 
                "lastAlert": "N/A",
                "confidence_score": 0.0,
                "model_threshold": GLOBAL_SETTINGS['fall_threshold']
            })
            # Check the actual thread status for reliability
            is_live_from_thread = cam_def.get('thread_instance') is not None and cam_def['thread_instance'].is_running
            
            cameras.append({
                "id": cam_id,
                "name": cam_def['name'],
                "source": str(cam_def['source']), # Ensure source is string for JSON serialization
                "isLive": is_live_from_thread, # Use thread state for reliability
                "status": status['status'],
                "color": status['color'],
                "lastAlert": status.get('lastAlert', 'N/A'),
                "confidence_score": status.get('confidence_score', 0.0), 
                "model_threshold": status.get('model_threshold', GLOBAL_SETTINGS['fall_threshold']) 
            })
    return jsonify({"success": True, "cameras": cameras})


# NEW ENDPOINT: Get all camera definitions (live and stopped)
@app.route('/api/cameras/all_definitions', methods=['GET'])
def api_get_all_definitions():
    if 'user_id' not in session: 
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    definitions = []
    with camera_lock:
        for cam_id, cam_def in CAMERA_DEFINITIONS.items():
            is_live_from_thread = cam_def.get('thread_instance') is not None and cam_def['thread_instance'].is_running
            definitions.append({
                "id": cam_id,
                "name": cam_def['name'],
                # Source is intentionally kept as string representation here for display
                "source": str(cam_def['source']), 
                "isLive": is_live_from_thread
            })
    return jsonify({"success": True, "definitions": definitions})


# NEW ENDPOINT: Restart an existing stopped camera
@app.route('/api/cameras/add_existing', methods=['POST'])
def api_add_existing_camera():
    if 'user_id' not in session: 
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if not camera_id:
        return jsonify({"success": False, "message": "Camera ID is required"}), 400

    with camera_lock:
        if camera_id not in CAMERA_DEFINITIONS:
            return jsonify({"success": False, "message": "Camera ID not found"}), 404
        
        cam_def = CAMERA_DEFINITIONS[camera_id]
        
        # Check if already live
        if cam_def.get('thread_instance') and cam_def['thread_instance'].is_running:
            return jsonify({"success": False, "message": "Camera is already live"}), 400
        
        # Restart the camera processor
        # Ensure 'source' is in the correct type (int or str)
        src_type = cam_def['source']
        try:
            # Check if source is a digit string and convert back to int
            if isinstance(src_type, str) and src_type.isdigit():
                 src_type = int(src_type)
        except:
             pass # Keep as string if conversion fails or it was already an int

        processor = CameraProcessor(camera_id=camera_id, src=src_type, name=cam_def['name'], device=device)
        processor.start()
        
        CAMERA_DEFINITIONS[camera_id]['isLive'] = True
        CAMERA_DEFINITIONS[camera_id]['thread_instance'] = processor
        
    return jsonify({"success": True, "message": f"Camera '{cam_def['name']}' restarted."})


@app.route('/api/cameras/add', methods=['POST'])
def api_add_camera():
    if 'user_id' not in session: return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    data = request.get_json()
    name = data.get('name')
    source_str = data.get('source')
    
    if not name or not source_str:
        return jsonify({"success": False, "message": "Name and source are required"}), 400

    try:
        # Try to convert source to integer for webcam index, otherwise keep as string (URL/path)
        source = int(source_str)
    except ValueError:
        source = source_str
    
    # Create unique ID for the camera
    camera_id = f"cam_{str(uuid.uuid4()).split('-')[0]}"
    
    # Start the processing thread
    processor = CameraProcessor(camera_id=camera_id, src=source, name=name, device=device)
    processor.start()
    
    # Store camera definition
    with camera_lock:
        CAMERA_DEFINITIONS[camera_id] = {
            "name": name, 
            "source": source, 
            "isLive": True,
            "thread_instance": processor
        }

    return jsonify({"success": True, "message": f"Camera '{name}' added.", "camera_id": camera_id})

# NEW ROUTE: Stop a running camera stream
@app.route('/api/cameras/stop/<camera_id>', methods=['POST'])
def api_stop_camera(camera_id):
    if 'user_id' not in session: return jsonify({"success": False, "message": "Unauthorized"}), 401

    if camera_id not in CAMERA_DEFINITIONS:
        return jsonify({"success": False, "message": "Camera ID not found"}), 404

    cam_def = CAMERA_DEFINITIONS[camera_id]
    processor = cam_def.get('thread_instance')

    if processor and processor.is_running:
        processor.is_running = False # Signal the thread to stop gracefully
        # Use threading.Thread's join for a clean stop
        processor.join(timeout=5) 

    with camera_lock:
        # Mark as offline regardless of successful thread join
        CAMERA_DEFINITIONS[camera_id]['isLive'] = False
        CAMERA_DEFINITIONS[camera_id]['thread_instance'] = None
        
        # Clean up status and shared frames immediately
        if camera_id in CAMERA_STATUS:
            del CAMERA_STATUS[camera_id]
        if camera_id in shared_frames:
            del shared_frames[camera_id]

    return jsonify({"success": True, "message": f"Camera '{cam_def['name']}' stopped and resources released."})


# FIXED: Completed the logic for video file uploads
@app.route('/api/cameras/upload', methods=['POST'])
def api_upload_camera():
    if 'user_id' not in session: return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    if 'video_file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
    
    file = request.files['video_file']
    name = request.form.get('name')

    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    if not name:
        return jsonify({"success": False, "message": "Camera name is required"}), 400 
    
    # 1. Save the file securely
    filename = secure_filename(file.filename)
    # Append a UUID to prevent collisions if users upload files with the same name
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    # 2. Create unique ID and use filepath as source
    camera_id = f"cam_{str(uuid.uuid4()).split('-')[0]}"
    name_safe = secure_filename(name) # Safely use the provided name

    # 3. Start the processing thread
    processor = CameraProcessor(camera_id=camera_id, src=filepath, name=name_safe, device=device)
    processor.start()
    
    # 4. Store camera definition
    with camera_lock:
        CAMERA_DEFINITIONS[camera_id] = {
            "name": name_safe, 
            "source": filepath, 
            "isLive": True,
            "thread_instance": processor
        }
    
    return jsonify({"success": True, "message": f"Video stream '{name_safe}' added.", "camera_id": camera_id})


# --- CRITICAL: STARTUP LOGIC ---
if __name__ == '__main__':
    # 1. Define a default camera to start immediately
    DEFAULT_CAMERA_ID = "main_webcam_0"
    DEFAULT_CAMERA_NAME = "Main Webcam Stream (Default)"
    DEFAULT_CAMERA_SOURCE = 0 # Use 0 for the default integrated webcam

    print(f"[{DEFAULT_CAMERA_NAME}] Attempting to start default camera...")
    
    # 2. Initialize and start the thread
    default_processor = CameraProcessor(
        camera_id=DEFAULT_CAMERA_ID, 
        src=DEFAULT_CAMERA_SOURCE, 
        name=DEFAULT_CAMERA_NAME,
        device=device
    )
    default_processor.start()

    # 3. Store the definition
    with camera_lock:
        CAMERA_DEFINITIONS[DEFAULT_CAMERA_ID] = {
            "name": DEFAULT_CAMERA_NAME,
            "source": DEFAULT_CAMERA_SOURCE,
            "isLive": True,
            "thread_instance": default_processor
        }

    # 4. Start the Flask application
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, threaded=True)