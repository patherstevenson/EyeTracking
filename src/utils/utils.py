#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`utils` module
====================

Utility functions for gaze tracking, including image processing, 
landmark extraction, and coordinate transformation.

:author: Pather Stevenson
:date: February 2025
"""

import cv2
import numpy as np
import scipy.io as sio
from typing import Optional, Tuple, Dict

from utils.config import SCREEN_WIDTH, SCREEN_HEIGHT, GAZE_RANGE_CM, MID_X, MID_Y

# MediaPipe marker IDs for facial landmarks
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
FACE_OVAL = list(range(10, 338))

def loadMetadata(filename: str, silent: bool = False) -> dict | None:
    """
    Loads metadata from a .mat file.

    :param filename: Path to the .mat file.
    :type filename: str
    :param silent: Whether to suppress print statements, defaults to False.
    :type silent: bool, optional
    :return: Dictionary containing metadata, or None if loading fails.
    :rtype: dict | None
    """
    try:
        if not silent:
            print(f"\tReading metadata from {filename}...")
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except Exception:
        print(f"\tFailed to read the meta file '{filename}'!")
        return None
    return metadata

def is_valid(bbox):
            x_min, y_min, x_max, y_max = bbox
            return (x_max - x_min) > 1 and (y_max - y_min) > 1

def get_bounding_box(indices: list[int], landmarks: list[tuple[int, int]],
                     width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT,
                      x_margin: int = 0, y_margin: int = 0) -> tuple[int, int, int, int]:
    """
    Computes the bounding box around specified facial landmarks.

    :param indices: List of landmark indices.
    :type indices: list[int]
    :param landmarks: List of (x, y) landmark coordinates.
    :type landmarks: list[tuple[int, int]]
    :param x_margin: Additional margin in the x-direction, defaults to 0.
    :type x_margin: int, optional
    :param y_margin: Additional margin in the y-direction, defaults to 0.
    :type y_margin: int, optional
    :return: Bounding box coordinates (x_min, y_min, x_max, y_max).
    :rtype: tuple[int, int, int, int]
    """
    coords = [landmarks[i] for i in indices]
    x_min = max(0, min(pt[0] for pt in coords) - x_margin)
    y_min = max(0, min(pt[1] for pt in coords) - y_margin)
    x_max = min(width, max(pt[0] for pt in coords) + x_margin)
    y_max = min(height, max(pt[1] for pt in coords) + y_margin)
    
    return x_min, y_min, x_max, y_max

def draw_bounding_boxes(frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None,
    left_eye_bbox: Optional[Tuple[int, int, int, int]] = None,
    right_eye_bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Draws the given bounding boxes on a webcam frame.

    :param frame: Captured image from the webcam (NumPy array).
    :param face_bbox: Optional; Tuple (x, y, w, h) representing the face bounding box.
    :param left_eye_bbox: Optional; Tuple (x, y, w, h) for the left eye bounding box.
    :param right_eye_bbox: Optional; Tuple (x, y, w, h) for the right eye bounding box.
    :return: The captured image with the bounding boxes drawn on it.
    """
    # Colors (B, G, R)
    FACE_COLOR = (0, 255, 0)   # Green
    EYE_COLOR = (255, 0, 0)    # Blue

    # Draw the bounding box for the face
    if face_bbox is not None:
        x, y, w, h = face_bbox
        cv2.rectangle(frame, (x, y), (w, h), FACE_COLOR, 2)

    # Draw the bounding box for the left eye
    if left_eye_bbox is not None:
        x, y, w, h = left_eye_bbox
        cv2.rectangle(frame, (x, y), (w, h), EYE_COLOR, 2)

    # Draw the bounding box for the right eye
    if right_eye_bbox is not None:
        x, y, w, h = right_eye_bbox
        cv2.rectangle(frame, (x, y), (w, h), EYE_COLOR, 2)

    return frame


def preprocess_roi(roi: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocesses an image region of interest (ROI) for model input.

    :param roi: Image region of interest.
    :type roi: np.ndarray
    :param size: Target size for resizing, defaults to (224, 224).
    :type size: tuple[int, int], optional
    :return: Preprocessed ROI image.
    :rtype: np.ndarray
    """
    roi_resized = cv2.resize(roi, size)
    roi_normalized = roi_resized / 255.0
    roi_prepared = np.expand_dims(roi_normalized, axis=0)
    return roi_prepared

def generate_face_grid(face_bbox: tuple[int, int, int, int], 
                       image_shape: tuple[int, int, int], grid_size: int = 25) -> np.ndarray:
    """
    Generates a face grid representation from the bounding box.

    :param face_bbox: Bounding box of the face (x_min, y_min, x_max, y_max).
    :type face_bbox: tuple[int, int, int, int]
    :param image_shape: Shape of the input image (height, width, channels).
    :type image_shape: tuple[int, int, int]
    :param grid_size: Size of the output face grid, defaults to 25.
    :type grid_size: int, optional
    :return: Flattened face grid array.
    :rtype: np.ndarray
    """
    face_grid = np.zeros((grid_size, grid_size))
    grid_x = int((face_bbox[0] / image_shape[1]) * grid_size)
    grid_y = int((face_bbox[1] / image_shape[0]) * grid_size)
    grid_w = int(((face_bbox[2] - face_bbox[0]) / image_shape[1]) * grid_size)
    grid_h = int(((face_bbox[3] - face_bbox[1]) / image_shape[0]) * grid_size)

    face_grid[grid_y:grid_y + grid_h, grid_x:grid_x + grid_w] = 1
    return face_grid.flatten()

def gaze_cm_to_pixels(gaze_x_cm: float, gaze_y_cm: float) -> tuple[int, int]:
    """
    Converts gaze prediction from cm (relative to center) into screen pixel coordinates.

    :param gaze_x_cm: X-coordinate of gaze in cm.
    :type gaze_x_cm: float
    :param gaze_y_cm: Y-coordinate of gaze in cm.
    :type gaze_y_cm: float
    :return: Pixel coordinates (x_pixel, y_pixel) on the screen.
    :rtype: tuple[int, int]
    """
    x_pixel = (GAZE_RANGE_CM + gaze_x_cm) / (2 * GAZE_RANGE_CM) * SCREEN_WIDTH
    y_pixel = (GAZE_RANGE_CM - gaze_y_cm) / (2 * GAZE_RANGE_CM) * SCREEN_HEIGHT

    return int(x_pixel), int(y_pixel)

def pixels_to_gaze_cm(x_pixel: int, y_pixel: int) -> tuple[float, float]:
    """
    Convert pixel coordinates to gaze coordinates in cm relative to screen center.

    :param x_pixel: X coordinate in pixels
    :param y_pixel: Y coordinate in pixels
    :return: (x_cm, y_cm) Gaze coordinates in cm
    """
    x_cm: float = ((x_pixel / SCREEN_WIDTH) * 2 * GAZE_RANGE_CM) - GAZE_RANGE_CM
    y_cm: float = GAZE_RANGE_CM - ((y_pixel / SCREEN_HEIGHT) * 2 * GAZE_RANGE_CM)

    return x_cm, y_cm

def euclidan_distance_radius(pt1: tuple[float, float], pt2: tuple[float, float], radius: float) -> bool:
    """
    Check if the Euclidean distance between two points is less than or equal to a given radius.

    :param pt1: (x1, y1) First point in cm
    :param pt2: (x2, y2) Second point in cm
    :param radius: Maximum allowable distance in cm
    :return: True if the distance between pt1 and pt2 is less than or equal to radius, otherwise False.
    """
    return np.linalg.norm(np.array(pt1) - np.array(pt2)) <= radius

def get_numbered_calibration_points() -> Dict[int, Tuple[int, int]]:
    """
    Returns the numbered calibration points as a dictionary {index: (x, y)}.
    
    :return: Dictionary with keys as indices (0-12) and values as (x, y) tuples.
    """
    return {
        0:  (int(SCREEN_WIDTH * 0.1) , int(SCREEN_HEIGHT * 0.1)),
        1:  (int(SCREEN_WIDTH * 0.25) , int(SCREEN_HEIGHT * 0.25)),
        2:  (int(SCREEN_WIDTH * 0.5) , int(SCREEN_HEIGHT * 0.1)),
        3:  (int(SCREEN_WIDTH * 0.75) , int(SCREEN_HEIGHT * 0.25)),
        4:  (int(SCREEN_WIDTH * 0.9) , int(SCREEN_HEIGHT * 0.1)),
        5:  (int(SCREEN_WIDTH * 0.1) , int(SCREEN_HEIGHT * 0.5)),
        6:  (MID_X, MID_Y),
        7:  (int(SCREEN_WIDTH * 0.9) , int(SCREEN_HEIGHT * 0.5)),
        8:  (int(SCREEN_WIDTH * 0.1) , int(SCREEN_HEIGHT * 0.9)),
        9:  (int(SCREEN_WIDTH * 0.25) , int(SCREEN_HEIGHT * 0.75)),
        10: (int(SCREEN_WIDTH * 0.5) , int(SCREEN_HEIGHT * 0.9)),
        11: (int(SCREEN_WIDTH * 0.75) , int(SCREEN_HEIGHT * 0.75)),
        12: (int(SCREEN_WIDTH * 0.9) , int(SCREEN_HEIGHT * 0.9)),
    }
