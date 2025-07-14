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
from typing import Optional, Tuple, Dict, Union
import torch
import gc
import mediapipe as mp
from tqdm import tqdm
import pandas as pd
import pickle
import os
from sklearn.model_selection import GroupKFold

from utils.config import SCREEN_WIDTH, SCREEN_HEIGHT, GAZE_RANGE_CM, MID_X, MID_Y

# MediaPipe marker IDs for facial landmarks
LEFT_EYE = [33, 133, 159, 160, 158, 144]
RIGHT_EYE = [362, 263, 386, 387, 385, 373]
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

def get_groupwise_train_test_split(
    df_clean: pd.DataFrame,
    subject_col: str = "subject",
    img_col: str = "img_path",
    gaze_cols: Tuple[str, str] = ("gaze_x", "gaze_y"),
    n_splits: int = 5,
    fold_index: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Effectue une séparation train/test groupée à l'aide de GroupKFold
    en garantissant qu'aucun sujet n'apparaît à la fois dans l'entraînement et le test.

    :param df_clean: DataFrame nettoyé contenant au moins les colonnes sujet, img_path et gaze.
    :param subject_col: Nom de la colonne identifiant les groupes (sujets).
    :param img_col: Nom de la colonne contenant les chemins d’images.
    :param gaze_cols: Tuple contenant les noms des colonnes de coordonnées du regard.
    :param n_splits: Nombre de splits (par défaut 5).
    :param fold_index: Index du split à utiliser (entre 0 et n_splits - 1).

    :return: Tuple contenant df_train et df_test pour le split spécifié.
    """
    groups = np.array(df_clean[subject_col])
    X = np.array(df_clean[img_col])
    y = np.array(df_clean[list(gaze_cols)])

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups))

    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError(f"fold_index doit être entre 0 et {n_splits - 1} (reçu {fold_index})")

    train_idx, test_idx = splits[fold_index]
    df_train = df_clean.iloc[train_idx].reset_index(drop=True)
    df_test = df_clean.iloc[test_idx].reset_index(drop=True)

    return df_train, df_test

def extract_inputs_from_image(face_mesh, img_path: str, means: dict) -> tuple | None:
    """
    Extracts facial features and normalizes them using provided means.

    :param face_mesh: Initialized MediaPipe face mesh detector.
    :param img_path: Path to the image.
    :param means: Dictionary of mean tensors for 'face', 'eye_left', 'eye_right'.
    :return: Tuple of normalized tensors (face, eye_left, eye_right, face_grid) or None.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image not found or unreadable")

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_mp = face_mesh.process(image_rgb)
        if not img_mp.multi_face_landmarks:
            raise ValueError("No face landmarks detected")

        landmarks = img_mp.multi_face_landmarks[0]
        h, w, _ = img.shape
        points = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks.landmark]

        # Bounding boxes
        left_eye_bbox = get_bounding_box(LEFT_EYE, points, w, h)
        right_eye_bbox = get_bounding_box(RIGHT_EYE, points, w, h)
        face_bbox = get_bounding_box(FACE_OVAL, points, w, h)

        # Preprocess ROIs
        left_eye_roi = preprocess_roi(img[left_eye_bbox[1]:left_eye_bbox[3], left_eye_bbox[0]:left_eye_bbox[2]])
        right_eye_roi = preprocess_roi(img[right_eye_bbox[1]:right_eye_bbox[3], right_eye_bbox[0]:right_eye_bbox[2]])
        face_roi = preprocess_roi(img[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]])

        # Convert to tensors and normalize with means
        face_tensor = torch.tensor(face_roi[0], dtype=torch.float32).permute(2, 0, 1).sub(means['face'])
        eye_left_tensor = torch.tensor(left_eye_roi[0], dtype=torch.float32).permute(2, 0, 1).sub(means['eye_left'])
        eye_right_tensor = torch.tensor(right_eye_roi[0], dtype=torch.float32).permute(2, 0, 1).sub(means['eye_right'])

        # Face grid
        face_grid = generate_face_grid(face_bbox, img.shape)
        face_grid_tensor = torch.tensor(face_grid, dtype=torch.float32).view(1, -1)

        return face_tensor, eye_left_tensor, eye_right_tensor, face_grid_tensor

    except Exception as e:
        try:
            with open("skipped_images.txt", "a") as f:
                f.write(f"[EXCEPTION] {img_path} -- {str(e)}\n")
                #if 'img' in locals():
                #    f.write(f"left_eye : {len(img[left_eye_bbox[1]:left_eye_bbox[3], left_eye_bbox[0]:left_eye_bbox[2]])} {left_eye_bbox}\n")
                #    f.write(f"right_eye: {len(img[right_eye_bbox[1]:right_eye_bbox[3], right_eye_bbox[0]:right_eye_bbox[2]])} {right_eye_bbox}\n")
                #    f.write(f"face     : {len(img[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]])} {face_bbox}\n")
        except Exception as logging_error:
            print(f"[WARNING] Could not log skipping info for {img_path}: {logging_error}")
        return None

def extract_and_save_batches(
    df: pd.DataFrame,
    prefix: str,
    output_dir: str,
    extract_fn=extract_inputs_from_image,
    batch_size: int = 1000,
    image_col: str = "img_path",
    gaze_cols: Union[tuple, list] = ("gaze_x", "gaze_y"),
    mat_dir: str = None,
):
    """
    Extracts MediaPipe features from a DataFrame and saves them in .pkl batches.

    :param df: DataFrame containing image paths and gaze coordinates
    :param output_dir: directory where .pkl batch files will be saved
    :param extract_fn: function to extract features from an image path
    :param batch_size: number of images to process per batch
    :param image_col: name of the column containing the image path
    :param gaze_cols: names of the columns containing gaze coordinates
    :param prefix: prefix for the saved batch filenames
    :param mat_dir: directory where the .mat files are located (default: src/mat)
    """

    os.makedirs(output_dir, exist_ok=True)

    if mat_dir is None:
        current_file = os.path.abspath(__file__)
        mat_dir = os.path.normpath(os.path.join(os.path.dirname(current_file), "..", "mat"))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1
    )

    meta_face = sio.loadmat(os.path.join(mat_dir, "mean_face_224_MPIIFace.mat"))
    meta_left = sio.loadmat(os.path.join(mat_dir, "mean_left_224_MPIIFace.mat"))
    meta_right = sio.loadmat(os.path.join(mat_dir, "mean_right_224_MPIIFace.mat"))

    means = {
        "face": torch.tensor(meta_face["mean_face"], dtype=torch.float32),
        "eye_left": torch.tensor(meta_left["mean_eye_left"], dtype=torch.float32),
        "eye_right": torch.tensor(meta_right["mean_eye_right"], dtype=torch.float32),
    }

    num_rows = len(df)
    num_batches = (num_rows + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_rows)
        
        df_batch = df.iloc[start:end]
        df_features_batch = []

        print(f"\n=== Processing batch {i+1}/{num_batches} ({start} to {end}) ===")

        for _, row in tqdm(df_batch.iterrows(), total=len(df_batch)):
            img_path = row[image_col]
            gaze = tuple(row[col] for col in gaze_cols)

            try:
                features = extract_fn(face_mesh, img_path, means)

                if features is not None:
                    face, eye_left, eye_right, face_grid = features

                    df_features_batch.append({
                        'img_path': img_path,
                        'gaze': gaze,
                        'face': face,
                        'eye_left': eye_left,
                        'eye_right': eye_right,
                        'face_grid': face_grid
                    })

            except Exception as e:
                print(f"[ERROR] Failed on {img_path}: {e}")
                continue

        batch_path = os.path.join(output_dir, f"{prefix}/batch_{prefix}_{i+1}.pkl")
        with open(batch_path, "wb") as f:
            pickle.dump(df_features_batch, f)

        print(f"[OK] Batch {prefix} {i+1} saved to {batch_path}")
        del df_features_batch
        gc.collect()