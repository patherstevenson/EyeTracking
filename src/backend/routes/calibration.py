from fastapi import APIRouter, Body
import cv2
import numpy as np
from src.backend.shared.state import gaze_tracker, capture_points, screen_width, screen_height
from src.tracker.CalibrationDataset import CalibrationDataset
from src.utils.utils import pixels_to_gaze_cm

router = APIRouter()

class FaceLandmarks:
    def __init__(self, landmarks):
        self.landmark = [Landmark(pt) for pt in landmarks]

class Landmark:
    def __init__(self, pt):
        self.x = pt["x"]
        self.y = pt["y"]
        self.z = pt["z"]

@router.post("/submit_calibration")
async def submit_calibration(payload: dict = Body(...)):
    """
    Accepts a list of 13 calibration points via JSON with base64-encoded images.
    Each point must include x_pixel, y_pixel, image_base64, and landmarks.
    """
    from base64 import b64decode
    import re

    try:
        points = payload["points"]
        if len(points) != 13:
            return {"error": "Exactly 13 points required."}

        capture_points.clear()

        for i, point in enumerate(points):
            x_pixel = point["x_pixel"]
            y_pixel = point["y_pixel"]
            landmarks_data = point["landmarks"]

            # Extract base64-encoded image
            base64_data = re.sub('^data:image/.+;base64,', '', point["image_base64"])
            image_bytes = b64decode(base64_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            face_landmarks = FaceLandmarks(landmarks_data)
            face_input, left_eye_input, right_eye_input, face_grid_input = gaze_tracker.extract_features(
                img, face_landmarks, screen_width, screen_height
            )
            x_cm, y_cm = pixels_to_gaze_cm(x_pixel, y_pixel, screen_width, screen_height)
            capture_points.append(((face_input, left_eye_input, right_eye_input, face_grid_input), (x_cm, y_cm)))

        gaze_tracker.train(CalibrationDataset(capture_points))
        return {"message": "Calibration and fine-tuning completed", "total_points": len(capture_points)}

    except Exception as e:
        return {"error": str(e)}
