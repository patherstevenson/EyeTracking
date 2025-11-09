#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`GazeTracker` module
=========================

This module handles gaze tracking using a deep learning model. It processes video input,
extracts facial landmarks, and predicts gaze direction. The module also logs gaze data 
for further analysis.

:author: Pather Stevenson
:date: February 2025
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import mediapipe as mp

from utils.utils import *
from utils.config import MID_X, MID_Y, LR, EPOCH, BATCH_SIZE

from .GazeDataLogger import GazeDataLogger
from .GazeModel import GazeModel
from .Calibration import Calibration

from OneEuroFilter import OneEuroFilter
import time


class GazeTracker:
    """
    Main class for gaze tracking.
    Manages model initialization and gaze prediction.
    """

    def __init__(self, model_path: str = None):
        """
        Initializes the gaze tracking system, loads the model, and prepares mean normalization values.

        :param model_path: Path to the pre-trained model checkpoint, defaults to None.
        :type model_path: str, optional
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mp = model_path
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models/{model_path}"))

        self._load_mean()
        self.model = GazeModel()
        self._load_model(self.model_path)
        self.logger = GazeDataLogger()
        self.margin: int = 20
        self.calibration = Calibration(self)
        self.window_name = "EyeTheia Live Gaze Visualization"

        self.gaze_filter_x = OneEuroFilter(
            freq=60,
            mincutoff=1.0,
            beta=0.007,     
            dcutoff=1.
        )
        self.gaze_filter_y = OneEuroFilter(
            freq=60,
            mincutoff=1.0,
            beta=0.007,
            dcutoff=1.0
        )

    def _load_model(self, model_path: str, verbose: bool = True) -> None:
        """
        Loads the gaze tracking model and its weights.

        :param model_path: Path to the model checkpoint.
        :param verbose: verbosity
        :type model_path: str
        :type verbose: bool
        """
        print(self.device)
        if verbose:
            match self.device:
                case "cuda":
                    print(f"\n-----------------\nDevice : {torch.cuda.get_device_name(torch.cuda.current_device())}\n-----------------\n")
                case "cpu":
                    print(f"\n-----------------\nDevice : CPU\n-----------------\n")
                case _:
                    pass

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _load_mean(self) -> None:
        """
        Loads mean images for normalization during preprocessing.
        """
        mean_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mat/"))

        match self.mp:
            case "itracker_baseline.tar":
                self.faceMean = loadMetadata(os.path.join(mean_path, 'mean_face_224.mat'))['image_mean']
                self.eyeLeftMean = loadMetadata(os.path.join(mean_path, 'mean_left_224.mat'))['image_mean']
                self.eyeRightMean = loadMetadata(os.path.join(mean_path, 'mean_right_224.mat'))['image_mean']
            case "itracker_mpiiface.tar":
                self.faceMean = loadMetadata(os.path.join(mean_path, 'mean_face_224_MPIIFace.mat'))['mean_face']
                self.eyeLeftMean = loadMetadata(os.path.join(mean_path, 'mean_left_224_MPIIFace.mat'))['mean_eye_left']
                self.eyeRightMean = loadMetadata(os.path.join(mean_path, 'mean_right_224_MPIIFace.mat'))['mean_eye_right']
            case _ :
                pass

        self.faceMean = torch.tensor(self.faceMean / 255.0, dtype=torch.float32)
        self.eyeLeftMean = torch.tensor(self.eyeLeftMean / 255.0, dtype=torch.float32)
        self.eyeRightMean = torch.tensor(self.eyeRightMean / 255.0, dtype=torch.float32)

    def save_tracking_data(self) -> None:
        """
        Saves logged gaze tracking data to a file.
        """
        self.logger.save_data()

    def reset_model(self) -> None:
        """
        Reloads the original pre-trained weights from disk.
        This is useful to reset the model before performing a new calibration,
        ensuring that fine-tuning does not accumulate across sessions.
        """
        print(f"[GazeTracker] Resetting model to original pre-trained weights ({self.mp}).")
        self._load_model(self.model_path, verbose=False)
        self.model.eval()

    def train(self, dataset: Dataset, epochs: int = 10, learning_rate: float = 1e-4, batch_size: int = 4) -> None:
        """
        Fine-tune the gaze tracking model with a calibration session

        :param dataset: A PyTorch Dataset containing new user-specific training samples.
        :param epochs: Number of epochs for fine-tuning.
        :param learning_rate: Learning rate for the optimizer.
        :param batch_size: Batch size for training.
        """
        self.model.train()
        self.model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for faces, eyes_left, eyes_right, face_grids, gaze_targets in dataloader:

                faces, eyes_left, eyes_right, face_grids, gaze_targets = (
                    faces.to(self.device),
                    eyes_left.to(self.device),
                    eyes_right.to(self.device),
                    face_grids.to(self.device),
                    gaze_targets.to(self.device)
                )

                optimizer.zero_grad()

                predictions = self.model(faces, eyes_left, eyes_right, face_grids)

                loss = criterion(predictions, gaze_targets)

                loss.backward()
                optimizer.step() 
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        print("Fine-tuning complete.")

    def extract_features(self, img: torch.Tensor, face_landmarks, SCREEN_WIDTH:int , SCREEN_HEIGHT: int) -> tuple:
        """
        Extracts facial features from the image using detected landmarks.

        :param img: Input frame from the webcam.
        :type img: torch.Tensor
        :param face_landmarks: Detected facial landmarks.
        :return: Preprocessed face, eye, and face grid tensors for model inference.
        :rtype: tuple
        """
        h, w, _ = img.shape
        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

        # Bounding boxes for eyes and face
        left_eye_bbox = get_bounding_box(LEFT_EYE, landmarks, SCREEN_WIDTH, SCREEN_HEIGHT, x_margin=self.margin, y_margin=self.margin)
        right_eye_bbox = get_bounding_box(RIGHT_EYE, landmarks, SCREEN_WIDTH, SCREEN_HEIGHT, x_margin=self.margin, y_margin=self.margin)
        face_bbox = get_bounding_box(FACE_OVAL, landmarks, SCREEN_WIDTH, SCREEN_HEIGHT, x_margin=self.margin, y_margin=self.margin)

        # Preprocess regions of interest
        left_eye_roi = preprocess_roi(img[left_eye_bbox[1]:left_eye_bbox[3], left_eye_bbox[0]:left_eye_bbox[2]])
        right_eye_roi = preprocess_roi(img[right_eye_bbox[1]:right_eye_bbox[3], right_eye_bbox[0]:right_eye_bbox[2]])
        face_roi = preprocess_roi(img[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]])

        # Generate inputs for the model
        match self.mp:
            case "itracker_mpiiface.tar":
                face_input = torch.tensor(face_roi[0], dtype=torch.float32).permute(2, 0, 1).sub(self.faceMean).to(self.device)
                left_eye_input = torch.tensor(left_eye_roi[0], dtype=torch.float32).permute(2, 0, 1).sub(self.eyeLeftMean).to(self.device)
                right_eye_input = torch.tensor(right_eye_roi[0], dtype=torch.float32).permute(2, 0, 1).sub(self.eyeRightMean).to(self.device)
                
                face_input = face_input.unsqueeze(0)
                left_eye_input = left_eye_input.unsqueeze(0)
                right_eye_input = right_eye_input.unsqueeze(0)
                
            case "itracker_baseline.tar":
                left_eye_input = torch.tensor(left_eye_roi, dtype=torch.float32).sub(self.eyeLeftMean).permute(0, 3, 1, 2).to(self.device)
                right_eye_input = torch.tensor(right_eye_roi, dtype=torch.float32).sub(self.eyeRightMean).permute(0, 3, 1, 2).to(self.device)
                face_input = torch.tensor(face_roi, dtype=torch.float32).sub(self.faceMean).permute(0, 3, 1, 2).to(self.device)
            
            case _:
                pass
      
        face_grid = generate_face_grid(face_bbox, img.shape)
        face_grid_input = torch.tensor(face_grid, dtype=torch.float32).view(1, -1).to(self.device)

        return face_input, left_eye_input, right_eye_input, face_grid_input

    def predict_gaze(self, face_input: torch.Tensor, left_eye_input: torch.Tensor,
                           right_eye_input: torch.Tensor, face_grid_input: torch.Tensor,) -> tuple[float, float]:
        """
        Predicts the gaze direction based on the input features.
        Converts the predicted gaze coordinates from centimeters to pixels
        and logs the result.

        :param face_input: Processed face image tensor.
        :type face_input: torch.Tensor
        :param left_eye_input: Processed left eye image tensor.
        :type left_eye_input: torch.Tensor
        :param right_eye_input: Processed right eye image tensor.
        :type right_eye_input: torch.Tensor
        :param face_grid_input: Face grid tensor representing spatial positioning.
        :type face_grid_input: torch.Tensor
        :return: Tuple containing cm coordinates (x, y)
        :rtype: tuple[float, float]
        """

        with torch.no_grad():
            gaze_prediction = self.model(
                face_input, left_eye_input, right_eye_input, face_grid_input
            )
            gaze_x, gaze_y = gaze_prediction.cpu().numpy().flatten()
            
            self.logger.log_data(gaze_x, gaze_y)

            return gaze_x, gaze_y

    def run(self, webcam: cv2.VideoCapture) -> None:
        """
        Runs the gaze tracking loop, capturing frames and processing gaze estimation.
        Performs calibration and fine-tuning before running the tracking.

        :param webcam: OpenCV VideoCapture object.
        :type webcam: cv2.VideoCapture
        """
        calibration_dataset = self.calibration.run_calibration(webcam)

        print("\nFine-tuning the model with calibration data...")
        self.train(calibration_dataset, epochs=EPOCH, learning_rate=LR, batch_size=BATCH_SIZE)

        # start eval of the processed calibration
        mean_error, std_error = self.calibration.evaluate_calibration_accuracy()

        # Setup fullscreen window for visualization
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        fullscreen = True

        # Default position in case we have no valid prediction yet
        gaze_x_px, gaze_y_px = MID_X, MID_Y
        smoothing_alpha = 0.2
        
        start_time = time.perf_counter()

        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
            while True:
                success, img = webcam.read()
                if not success:
                    print("Error reading from the webcam.")
                    break

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_mp = face_mesh.process(img_rgb)

                if img_mp.multi_face_landmarks:
                    for face_landmarks in img_mp.multi_face_landmarks:
                        face_input, left_eye_input, right_eye_input, face_grid_input = self.extract_features(
                            img, face_landmarks, SCREEN_WIDTH, SCREEN_HEIGHT
                        )

                        gaze_x, gaze_y = self.predict_gaze(
                            face_input, left_eye_input, right_eye_input, face_grid_input
                        )

                        match self.mp:
                            case "itracker_mpiiface.tar":
                                gx_px, gy_px = denormalized_MPIIFaceGaze(
                                    gaze_x, gaze_y, SCREEN_WIDTH, SCREEN_HEIGHT
                                )
                            case "itracker_baseline.tar":
                                gx_px, gy_px = gaze_cm_to_pixels(
                                    gaze_x, gaze_y, SCREEN_WIDTH, SCREEN_HEIGHT
                                )
                            case _:
                                raise ValueError("invalid model_path")

                        # --- One Euro Filter smoothing (timestamp in seconds) ---
                        timestamp = time.perf_counter() - start_time
                        gx_px = float(self.gaze_filter_x.filter(float(gx_px), timestamp))
                        gy_px = float(self.gaze_filter_y.filter(float(gy_px), timestamp))

                        # store last filtered values
                        gaze_x_px = prev_x + smoothing_alpha * (gx_px - prev_x)
                        gaze_y_px = prev_y + smoothing_alpha * (gy_px - prev_y)

                        # update previous position for next frame
                        prev_x, prev_y = gaze_x_px, gaze_y_px
                        
                # White fullscreen background
                white_bg = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 255

                # Draw solid black dot at filtered gaze position
                cx, cy = int(gaze_x_px), int(gaze_y_px)
                cv2.circle(white_bg, (cx, cy), 10, (0, 0, 0), -1)

                # (Optionnel) hint utilisateur
                cv2.putText(
                    white_bg,
                    "Esc: toggle fullscreen  |  Q: quit",
                    (20, SCREEN_HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(self.window_name, white_bg)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == 27:  # ESC
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        self.window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )

        cv2.destroyWindow(self.window_name)
        #self.logger.save_data()
