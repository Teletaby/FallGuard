import time
import base64
import logging
import cv2
import numpy as np
from flask import Flask, request, jsonify 

# Import your core logic from sibling files (pose_estimator.py)
try:
    from .pose_estimator import PoseStreamProcessor
except ImportError:
    logging.warning("Failed to import PoseStreamProcessor as a module. Assuming a local run setup.")
    from pose_estimator import PoseStreamProcessor 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application and Utility Class Definitions

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


# Initialize the Flask application
app = Flask(__name__) 

try:
    pose_processor = PoseStreamProcessor()
    logger.info("PoseStreamProcessor initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize PoseStreamProcessor: {e}")

# Global dictionary to track timers for different cameras/sessions
camera_timers = {}

# API Endpoints

@app.route('/')
def health_check():
    """Confirms the service is running (Your current working route)."""
    return 'Fall Detection Service is Running'


@app.route('/detect', methods=['POST'])
def detect_fall_api():
    """
    Receives frame data, processes it, runs fall logic, and returns a result.
    """
    try:
        data = request.get_json()
        
        # Expecting a base64 encoded image string
        base64_frame = data.get('frame_bytes_b64')
        camera_id = data.get('camera_id', 'default_camera')
        frame_counter = data.get('frame_counter', 0)
        
        if not base64_frame:
            return jsonify({"error": "Missing 'frame_bytes_b64' data"}), 400

        # Decode the base64 string back to raw image bytes
        frame_bytes = base64.b64decode(base64_frame)

        results = pose_processor.process_frame_bytes(
            frame_bytes=frame_bytes, 
            frame_counter=frame_counter,
            camera_index=camera_id
        )

        if camera_id not in camera_timers:
            camera_timers[camera_id] = FallTimer()

        is_falling_pose = False
        fall_detected = False

        if results and results.get('landmarks'):
            fall_detected = camera_timers[camera_id].update(is_falling_pose)
        
        if fall_detected:
            # TODO: Trigger Pushbullet notification or other alert here
            logger.warning(f"!!! FALL DETECTED on camera {camera_id} !!!")
            camera_timers[camera_id].reset_camera_history(camera_id)


        return jsonify({
            "status": "processed", 
            "camera_id": camera_id,
            "fall_detected": fall_detected,
            "is_currently_falling_pose": is_falling_pose,
            "landmarks_count": len(results.get('landmarks', [])) if results else 0
        }), 200

    except ValueError as ve:
        logger.error(f"Processing Error for {camera_id}: {ve}")
        return jsonify({"error": str(ve)}), 500
        
    except Exception as e:
        logger.error(f"Unexpected Server Error: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == '__main__':
    app.run(debug=True)