from fastapi import APIRouter, HTTPException, Form
from src.tracker.CalibrationDataset import CalibrationDataset
from src.backend.shared.state import capture_points, gaze_tracker, screen_height, screen_width

import base64
import cv2
import numpy as np
import json
from src.backend.routes.calibration import FaceLandmarks

router = APIRouter()

@router.post("/fine_tune")
async def fine_tune_model():
    """
    Fine-tune le modèle `GazeTracker` avec les données de calibration stockées dans `capture_points`.
    """
    if not capture_points:
        raise HTTPException(status_code=400, detail="No calibration data available for fine-tuning.")

    try:
        gaze_tracker.train(CalibrationDataset(capture_points))

        return {"message": "Model fine-tuned successfully", "total_points": len(capture_points)}

    except Exception as e:
        return {"error": str(e)}

@router.post("/predict_gaze")
async def predict_gaze(
    image_base64: str = Form(...),
    landmarks: str = Form(...)
):
    try:
        # Decode image
        image_data = image_base64.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Parse landmarks
        landmarks_data = json.loads(landmarks)
        face_landmarks = FaceLandmarks(landmarks_data)

        # Predict gaze
        face_input, left_eye_input, right_eye_input, face_grid_input = gaze_tracker.extract_features(img, face_landmarks, screen_width, screen_height)
        gaze_cm = gaze_tracker.predict_gaze(face_input, left_eye_input, right_eye_input, face_grid_input)
        print(gaze_cm)
        return {"x_cm": gaze_cm[0], "y_cm": gaze_cm[1]}
    
    except Exception as e:
        return {"error": str(e)}