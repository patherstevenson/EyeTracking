#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`model` module
====================

This module defines model-related endpoints for real-time gaze prediction.
It decodes incoming frames and landmarks, extracts features via the injected
GazeTracker instance, predicts the gaze in calibrated space, and converts
the result back to screen pixels.

Input payload uses multipart/form-data (Form) to support large base64 images.

:author: Pather Stevenson
:date: October 2025
"""
from fastapi import APIRouter, HTTPException, Form
from utils import denormalized_MPIIFaceGaze
from routes.calibration import FaceLandmarks
from fastapi import APIRouter, HTTPException, Form, Depends
import base64
import cv2
import numpy as np
import json
import re

from routes.dependency import get_tracker, get_screen
from utils.utils import denormalized_MPIIFaceGaze, gaze_cm_to_pixels

router = APIRouter()

@router.post("/predict_gaze")
async def predict_gaze(
    image_base64: str = Form(...),
    landmarks: str = Form(...),
    gaze_tracker=Depends(get_tracker),
    screen=Depends(get_screen),
):
    """
    Perform a single-frame gaze prediction.

    Expected form fields:
      - image_base64 (str): data-URI or raw base64 image (RGB/BGR). Example: "data:image/png;base64,...."
      - landmarks (str): JSON-encoded list[dict] of facial landmarks with keys {x, y, z}.

    Returns:
      JSON with predicted gaze in pixels:
        { "x_px": float, "y_px": float }
    """
    try:
        screen_width, screen_height = screen

        # Decode image from base64 (support data URI or plain base64)
        # Strip potential data URL prefix like "data:image/png;base64,..."
        payload_base64 = re.sub(r"^data:image/.+;base64,", "", image_base64)
        img_bytes = base64.b64decode(payload_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image payload")

        # Parse and wrap landmarks
        try:
            landmarks_data = json.loads(landmarks)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid landmarks JSON: {e}") from e
        face_landmarks = FaceLandmarks(landmarks_data)

        # Extract model inputs
        face_input, left_eye_input, right_eye_input, face_grid_input = gaze_tracker.extract_features(
            img, face_landmarks, screen_width, screen_height
        )

        # Predict in the model's calibrated space (e.g., normalized for MPIIFaceGaze)
        predict_gaze = gaze_tracker.predict_gaze(face_input, left_eye_input, right_eye_input, face_grid_input)
        if not isinstance(predict_gaze, (list, tuple)) or len(predict_gaze) < 2:
            raise HTTPException(status_code=500, detail="Model returned invalid gaze vector")

        # Convert prediction back to screen pixels
        match gaze_tracker.mp:
            case "itracker_baseline.tar":
                x_px, y_px = gaze_cm_to_pixels(predict_gaze[0], predict_gaze[1], screen_width, screen_height)
            case "itracker_mpiiface.tar":
                x_px, y_px = denormalized_MPIIFaceGaze(predict_gaze[0], predict_gaze[1], screen_width, screen_height)
            case _:
                raise ValueError("Invalid model path")

        return {"x_px": float(x_px), "y_px": float(y_px)}

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions unchanged
        raise
    except Exception as e:
        # Generic safety net for unexpected errors
        return {"error": str(e)}
