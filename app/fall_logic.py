import time
import base64
import logging
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from flask import Flask, request, jsonify 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose components globally
mp_pose = mp.solutions.pose

# ----------------------------------------------------
# 1. UTILITY CLASS DEFINITIONS (FallTimer)
# ----------------------------------------------------

class FallTimer:
    def __init__(self, threshold=10):
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
# 2. CORE LOGIC CLASS DEFINITION (PoseStreamProcessor)
#    (Pasted directly into this file to avoid import errors)
# ----------------------------------------------------

class PoseStreamProcessor:
    def __init__(self):
        # MediaPipe Pose initialization (as per your code)
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
        logger.info("MediaPipe Pose initialized within fall_logic.py.")

    def _smooth_landmarks(self, camera_index: int, landmarks: list[dict]) -> list[dict]:
        # Full smoothing logic goes here (use the method from your pose_estimator.py content)
        # This is the implementation you provided:
        # ----------------------------------------------------------------------------------
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
        # Weights for temporal smoothing
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
        # ----------------------------------------------------------------------------------

    # NOTE: You must also paste in the _is_valid_human_pose method here for completeness!
    def _is_valid_human_pose(self, landmarks: list[dict]) -> bool:
        # Implementation from your provided pose_estimator.py content
        # ... (Pasted here)
        if not landmarks or len(landmarks) < 33:
            return False
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        key_landmarks = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        visible_key_landmarks = sum(1 for idx in key_landmarks if landmarks[idx].get("visibility", 0) > 0.5)
        if visible_key_landmarks < 2: return False
        
        try:
            left_shoulder_pos = landmarks[LEFT_SHOULDER]
            right_shoulder_pos = landmarks[RIGHT_SHOULDER]
            left_hip_pos = landmarks[LEFT_HIP]
            right_hip_pos = landmarks[RIGHT_HIP]
            shoulder_width = abs(left_shoulder_pos["x"] - right_shoulder_pos["x"])
            torso_height = abs(((left_shoulder_pos["y"] + right_shoulder_pos["y"]) / 2) - ((left_hip_pos["y"] + right_hip_pos["y"]) / 2))
            if shoulder_width < 0.02 or torso_height < 0.02: return False
            torso_ratio = torso_height / shoulder_width if shoulder_width > 0 else 0
            if torso_ratio < 0.3 or torso_ratio > 6.0: return False
        except (KeyError, ZeroDivisionError): pass
        
        try:
            visible_landmarks = [lm for lm in landmarks if lm.get("visibility", 0) > 0.4]
            if len(visible_landmarks) < 8: return False
            x_coords = [lm["x"] for lm in visible_landmarks]
            y_coords = [lm["y"] for lm in visible_landmarks]
            bbox_width = max(x_coords) - min(x_coords)
            bbox_height = max(y_coords) - min(y_coords)
            if bbox_width < 0.03 or bbox_height < 0.05: return False
        except (ValueError, ZeroDivisionError): return False
        
        return True
        # ----------------------------------------------------------------------------------

    # The async definition must be changed to sync since Flask is sync unless you use Quart
    def process_frame_bytes(
        self, 
        frame_bytes: bytes, 
        frame_counter: int,
        camera_index: int = 0
    ) -> dict | None:
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode image bytes.")
            
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
            logger.error(f"Pose processing error: {e}", exc_info=True)
            raise ValueError("Could not process pose estimation.")
    
    def reset_camera_history(self, camera_index: int):
        if camera_index in self.landmark_history:
            self.landmark_history[camera_index].clear()

# ----------------------------------------------------
# 3. FLASK APPLICATION INITIALIZATION & ENDPOINTS
# ----------------------------------------------------
app = Flask(__name__) 

# Initialize processor globally
try:
    pose_processor = PoseStreamProcessor()
except Exception as e:
    logger.error(f"Critical Error initializing PoseStreamProcessor: {e}")
    pose_processor = None # Set to None to prevent calls

camera_timers = {}

@app.route('/')
def health_check():
    return 'Fall Detection Service is Running'


@app.route('/detect', methods=['POST'])
def detect_fall_api():
    if pose_processor is None:
         return jsonify({"error": "Service not initialized"}), 503
         
    try:
        data = request.get_json()
        base64_frame = data.get('frame_bytes_b64')
        camera_id = data.get('camera_id', 'default_camera')
        frame_counter = data.get('frame_counter', 0)
        
        if not base64_frame:
            return jsonify({"error": "Missing 'frame_bytes_b64' data"}), 400

        frame_bytes = base64.b64decode(base64_frame)
        
        results = pose_processor.process_frame_bytes(
            frame_bytes=frame_bytes, 
            frame_counter=frame_counter,
            camera_index=camera_id
        )
        
        if camera_id not in camera_timers:
            camera_timers[camera_id] = FallTimer(threshold=10) 

        is_falling_pose = False
        fall_detected = False
        landmarks_count = 0

        if results and results.get('landmarks'):
            landmarks_count = len(results['landmarks'])
            # TODO: Integrate your actual ML/RNN model here
            # For testing: is_falling_pose = True # (to test the timer)
            
            fall_detected = camera_timers[camera_id].update(is_falling_pose)
        
        if fall_detected:
            # TODO: Trigger Pushbullet notification
            logger.warning(f"!!! CONFIRMED FALL DETECTED on camera {camera_id} !!!")
            pose_processor.reset_camera_history(camera_id)
            camera_timers[camera_id].reset_camera_history(camera_id)


        return jsonify({
            "status": "processed", 
            "camera_id": camera_id,
            "fall_detected": fall_detected,
            "landmarks_count": landmarks_count
        }), 200

    except Exception as e:
        logger.error(f"API Request Error: {e}", exc_info=True)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)