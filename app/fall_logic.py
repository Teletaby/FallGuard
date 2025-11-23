import time
import base64
import logging
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from flask import Flask, request, jsonify, send_from_directory, Response
import os
import threading
from datetime import datetime
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose components globally
mp_pose = mp.solutions.pose

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
        
    def reset_camera_history(self, camera_index: str):
        self.start_time = None

# ----------------------------------------------------
# POSE PROCESSOR
# ----------------------------------------------------

class PoseStreamProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.landmark_history = {}
        self.history_size = 5
        logger.info("MediaPipe Pose initialized.")

    def _smooth_landmarks(self, camera_index: int, landmarks: list[dict]) -> list[dict]:
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
        if visible_key_landmarks < 2: 
            return False
        
        try:
            left_shoulder_pos = landmarks[LEFT_SHOULDER]
            right_shoulder_pos = landmarks[RIGHT_SHOULDER]
            left_hip_pos = landmarks[LEFT_HIP]
            right_hip_pos = landmarks[RIGHT_HIP]
            shoulder_width = abs(left_shoulder_pos["x"] - right_shoulder_pos["x"])
            torso_height = abs(((left_shoulder_pos["y"] + right_shoulder_pos["y"]) / 2) - ((left_hip_pos["y"] + right_hip_pos["y"]) / 2))
            if shoulder_width < 0.02 or torso_height < 0.02: 
                return False
            torso_ratio = torso_height / shoulder_width if shoulder_width > 0 else 0
            if torso_ratio < 0.3 or torso_ratio > 6.0: 
                return False
        except (KeyError, ZeroDivisionError): 
            pass
        
        return True

    def process_frame(self, frame, camera_index: int = 0):
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
                    "x": landmark.x, "y": landmark.y, "z": landmark.z, "visibility": landmark.visibility
                })
            
            if not self._is_valid_human_pose(landmarks):
                return None
            
            smoothed_landmarks = self._smooth_landmarks(camera_index, landmarks)
            return {"landmarks": smoothed_landmarks}
        
        except Exception as e:
            logger.error(f"Pose processing error: {e}")
            return None
    
    def reset_camera_history(self, camera_index: int):
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
        self.settings = {
            'fall_threshold': 0.95,
            'fall_delay_seconds': 3
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
            'confidence_score': 0.0,
            'model_threshold': self.settings['fall_threshold']
        }
        self.camera_locks[camera_id] = threading.Lock()
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
        logger.info(f"Camera removed: {camera_id}")

# ----------------------------------------------------
# FLASK APPLICATION
# ----------------------------------------------------

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize components
pose_processor = PoseStreamProcessor()
camera_manager = CameraManager()
camera_timers = {}

# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        data = request.get_json()
        camera_manager.settings['fall_threshold'] = data.get('fall_threshold', 0.95)
        camera_manager.settings['fall_delay_seconds'] = data.get('fall_delay_seconds', 3)
        
        for cam_id in camera_manager.cameras:
            camera_manager.cameras[cam_id]['model_threshold'] = camera_manager.settings['fall_threshold']
        
        return jsonify({'success': True, 'settings': camera_manager.settings})
    
    return jsonify({'settings': camera_manager.settings})

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    cameras = list(camera_manager.cameras.values())
    return jsonify({'cameras': cameras})

@app.route('/api/cameras/all_definitions', methods=['GET'])
def get_all_definitions():
    definitions = list(camera_manager.cameras.values())
    return jsonify({'definitions': definitions})

@app.route('/api/cameras/add', methods=['POST'])
def add_camera():
    data = request.get_json()
    name = data.get('name')
    source = data.get('source')
    
    if not name or not source:
        return jsonify({'success': False, 'message': 'Name and source required'}), 400
    
    camera_id = f"camera_{len(camera_manager.cameras)}_{int(time.time())}"
    camera_manager.add_camera(camera_id, name, source)
    
    # For URLs, try to start immediately
    if source.startswith('http://') or source.startswith('https://') or source.startswith('rtsp://'):
        camera_manager.start_camera(camera_id)
    
    return jsonify({'success': True, 'camera_id': camera_id})

@app.route('/api/cameras/add_existing', methods=['POST'])
def add_existing_camera():
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if camera_id in camera_manager.cameras:
        camera_manager.start_camera(camera_id)
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'message': 'Camera not found'}), 404

@app.route('/api/cameras/stop/<camera_id>', methods=['POST'])
def stop_camera(camera_id):
    camera_manager.stop_camera(camera_id)
    return jsonify({'success': True})

@app.route('/api/cameras/remove/<camera_id>', methods=['DELETE'])
def remove_camera(camera_id):
    camera_manager.remove_camera(camera_id)
    return jsonify({'success': True})

@app.route('/api/cameras/upload', methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400
    
    file = request.files['video_file']
    name = request.form.get('name', 'Uploaded Video')
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    # Secure the filename
    filename = secure_filename(file.filename)
    filename = f"video_{int(time.time())}_{filename}"
    filepath = os.path.join(camera_manager.upload_folder, filename)
    
    try:
        file.save(filepath)
        logger.info(f"Video saved: {filepath}")
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return jsonify({'success': False, 'message': 'Failed to save file'}), 500
    
    # Add as camera source
    camera_id = f"video_{int(time.time())}"
    camera_manager.add_camera(camera_id, name, filepath)
    camera_manager.start_camera(camera_id)
    
    return jsonify({'success': True, 'camera_id': camera_id, 'message': 'Video uploaded successfully'})

def generate_frames(camera_id):
    """Generator function for video streaming"""
    fps_limit = 15  # Limit FPS for cloud deployment
    frame_delay = 1.0 / fps_limit
    
    while True:
        start_time = time.time()
        frame = camera_manager.get_frame(camera_id)
        
        if frame is None:
            # Send a blank frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, 'No Signal', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
        else:
            # Process frame for fall detection
            if camera_id not in camera_timers:
                camera_timers[camera_id] = FallTimer(threshold=camera_manager.settings['fall_delay_seconds'])
            
            pose_result = pose_processor.process_frame(frame, camera_id)
            
            # TODO: Add your ML model prediction here
            is_falling = False  # Replace with actual model prediction
            
            if pose_result:
                # Optional: Draw skeleton on frame
                pass
            
            fall_detected = camera_timers[camera_id].update(is_falling)
            
            if fall_detected:
                camera_manager.cameras[camera_id]['color'] = 'red'
                logger.warning(f"⚠️ FALL DETECTED on {camera_id}")
                # TODO: Send Pushbullet notification here
            else:
                camera_manager.cameras[camera_id]['color'] = 'green'
            
            # Update confidence score
            camera_manager.cameras[camera_id]['confidence_score'] = 0.5 if pose_result else 0.0
            
            # Resize frame for lower bandwidth
            frame = cv2.resize(frame, (640, 480))
            
            # Encode frame with compression
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
        
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
    return jsonify({'status': 'healthy', 'cameras': len(camera_manager.cameras)}), 200

# ----------------------------------------------------
# ERROR HANDLERS
# ----------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ----------------------------------------------------
# RUN
# ----------------------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)