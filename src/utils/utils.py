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
from utils.config import SCREEN_WIDTH, SCREEN_HEIGHT, GAZE_RANGE_CM

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

def get_bounding_box(indices: list[int], landmarks: list[tuple[int, int]], 
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
    x_max = min(SCREEN_WIDTH, max(pt[0] for pt in coords) + x_margin)
    y_max = min(SCREEN_HEIGHT, max(pt[1] for pt in coords) + y_margin)
    return x_min, y_min, x_max, y_max

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
