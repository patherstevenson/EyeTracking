#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`calibration` module
==========================

This module defines the calibration endpoints for gaze tracking.
It receives calibration points with corresponding images and landmarks,
extracts features using the current GazeTracker instance, normalizes the 
coordinates, and performs fine-tuning with the collected dataset.

:author: Pather Stevenson
:date: February 2025
"""

from fastapi import APIRouter, Body, Depends, HTTPException
import cv2
import numpy as np
import re
from base64 import b64decode

from tracker.CalibrationDataset import CalibrationDataset
from utils.utils import pixels_to_gaze_cm, normalize_MPIIFaceGaze
from routes.dependency import get_tracker, get_capture_points, get_screen

router = APIRouter()


class FaceLandmarks:
    """Helper wrapper for MediaPipe landmarks."""
    def __init__(self, landmarks):
        self.landmark = [Landmark(pt) for pt in landmarks]


class Landmark:
    """Single landmark point (x, y, z)."""
    def __init__(self, pt):
        self.x = pt["x"]
        self.y = pt["y"]
        self.z = pt["z"]


@router.post("/submit_calibration")
async def submit_calibration(
    payload: dict = Body(...),
    gaze_tracker=Depends(get_tracker),
    capture_points=Depends(get_capture_points),
    screen=Depends(get_screen),
):
    """
    Accepts a list of 17 calibration points via JSON with base64-encoded images.
    Each point must include:
      - x_pixel, y_pixel (screen coordinates),
      - image_base64 (string),
      - landmarks (face + eyes landmarks).
    """
    try:
        points = payload["points"]
        if len(points) != 17:
            return {"error": "Exactly 17 points required."}

        screen_width, screen_height = screen
        capture_points.clear()

        for i, point in enumerate(points):
            x_pixel = point["x_pixel"]
            y_pixel = point["y_pixel"]
            landmarks_data = point["landmarks"]

            # Decode base64-encoded image
            base64_data = re.sub(r"^data:image/.+;base64,", "", point["image_base64"])
            img_bytes = b64decode(base64_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Extract features
            face_landmarks = FaceLandmarks(landmarks_data)
            face_input, left_eye_input, right_eye_input, face_grid_input = gaze_tracker.extract_features(
                img, face_landmarks, screen_width, screen_height
            )

            # Normalize calibration coordinates depending on the model type
            match gaze_tracker.mp:
                case "itracker_baseline.tar":
                    x, y = pixels_to_gaze_cm(x_pixel, y_pixel, screen_width, screen_height)
                case "itracker_mpiiface.tar":
                    x, y = normalize_MPIIFaceGaze(x_pixel, y_pixel, screen_width, screen_height)
                case _:
                    raise ValueError("Invalid model path")

            capture_points.append(
                ((face_input, left_eye_input, right_eye_input, face_grid_input), (x, y))
            )

        # Reload model weights before fine-tuning
        gaze_tracker.reset_model()

        # Evaluate and fine-tune
        gaze_tracker.calibration.set_capture_points(capture_points)

        print("[Calibration] Before fine-tuning")
        gaze_tracker.calibration.evaluate_calibration_accuracy()

        gaze_tracker.train(CalibrationDataset(capture_points))

        print("[Calibration] After fine-tuning")
        gaze_tracker.calibration.evaluate_calibration_accuracy()

        return {"message": "Calibration and fine-tuning completed", "total_points": len(capture_points)}

    except Exception as e:
        return {"error": str(e)}
