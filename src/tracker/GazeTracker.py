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

        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/checkpoint.pth.tar"))

        self._load_mean()
        self.model = GazeModel()
        self._load_model(model_path)
        self.logger = GazeDataLogger()
        self.margin: int = 20
        self.calibration = Calibration(self)

    def _load_model(self, model_path: str) -> None:
        """
        Loads the gaze tracking model and its weights.

        :param model_path: Path to the model checkpoint.
        :type model_path: str
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _load_mean(self) -> None:
        """
        Loads mean images for normalization during preprocessing.
        """
        mean_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mat/"))

        self.faceMean = loadMetadata(os.path.join(mean_path, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(mean_path, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(mean_path, 'mean_right_224.mat'))['image_mean']

        self.faceMean = torch.tensor(self.faceMean / 255.0, dtype=torch.float32)
        self.eyeLeftMean = torch.tensor(self.eyeLeftMean / 255.0, dtype=torch.float32)
        self.eyeRightMean = torch.tensor(self.eyeRightMean / 255.0, dtype=torch.float32)

    def _determine_position(self, pos_x: int, pos_y: int) -> str:
        """
        Determines the screen position of the gaze based on pixel coordinates.

        :param pos_x: X-coordinate of the gaze position in pixels.
        :type pos_x: int
        :param pos_y: Y-coordinate of the gaze position in pixels.
        :type pos_y: int
        :return: Position quadrant of the gaze.
        :rtype: str
        """
        if pos_x < MID_X and pos_y < MID_Y:
            return "Top Left"
        elif pos_x > MID_X and pos_y < MID_Y:
            return "Top Right"
        elif pos_x < MID_X and pos_y > MID_Y:
            return "Bottom Left"
        elif pos_x > MID_X and pos_y > MID_Y:
            return "Bottom Right"
        else:
            return "Center"

    def _determine_quadrant(self, gaze_x: float, gaze_y: float) -> str:
        """
        Determines the gaze quadrant based on model output in centimeters.

        :param gaze_x: Predicted gaze X-coordinate in cm.
        :type gaze_x: float
        :param gaze_y: Predicted gaze Y-coordinate in cm.
        :type gaze_y: float
        :return: Gaze quadrant.
        :rtype: str
        """
        if gaze_x < 0 and gaze_y > 0:
            return "Top Left"
        elif gaze_x > 0 and gaze_y > 0:
            return "Top Right"
        elif gaze_x < 0 and gaze_y < 0:
            return "Bottom Left"
        elif gaze_x > 0 and gaze_y < 0:
            return "Bottom Right"
        else:
            return "Center"

    def save_tracking_data(self) -> None:
        """
        Saves logged gaze tracking data to a file.
        """
        self.logger.save_data()

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


    def extract_features(self, img: torch.Tensor, face_landmarks) -> tuple:
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
        left_eye_bbox = get_bounding_box(LEFT_EYE, landmarks, x_margin=self.margin, y_margin=self.margin)
        right_eye_bbox = get_bounding_box(RIGHT_EYE, landmarks, x_margin=self.margin, y_margin=self.margin)
        face_bbox = get_bounding_box(FACE_OVAL, landmarks, x_margin=self.margin, y_margin=self.margin)

        # Preprocess regions of interest
        left_eye_roi = preprocess_roi(img[left_eye_bbox[1]:left_eye_bbox[3], left_eye_bbox[0]:left_eye_bbox[2]])
        right_eye_roi = preprocess_roi(img[right_eye_bbox[1]:right_eye_bbox[3], right_eye_bbox[0]:right_eye_bbox[2]])
        face_roi = preprocess_roi(img[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]])

        # Generate inputs for the model
        left_eye_input = torch.tensor(left_eye_roi, dtype=torch.float32).sub(self.eyeLeftMean).permute(0, 3, 1, 2).to(self.device)
        right_eye_input = torch.tensor(right_eye_roi, dtype=torch.float32).sub(self.eyeRightMean).permute(0, 3, 1, 2).to(self.device)

        face_grid = generate_face_grid(face_bbox, img.shape)
        face_input = torch.tensor(face_roi, dtype=torch.float32).sub(self.faceMean).permute(0, 3, 1, 2).to(self.device)
        face_grid_input = torch.tensor(face_grid, dtype=torch.float32).view(1, -1).to(self.device)

        return face_input, left_eye_input, right_eye_input, face_grid_input

    def predict_gaze(self, face_input: torch.Tensor, left_eye_input: torch.Tensor,
                           right_eye_input: torch.Tensor, face_grid_input: torch.Tensor,) -> tuple[int, int, str]:
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
        :return: Tuple containing pixel coordinates (x, y) and the screen position.
        :rtype: tuple[int, int, str]
        """

        with torch.no_grad():
            gaze_prediction = self.model(
                face_input, left_eye_input, right_eye_input, face_grid_input
            )
            gaze_x, gaze_y = gaze_prediction.cpu().numpy().flatten()

            # Convert to pixel coordinates
            pos_x, pos_y = gaze_cm_to_pixels(gaze_x, gaze_y)

            # Determine the position on the screen
            position = self._determine_position(pos_x, pos_y)
            quadrant = self._determine_quadrant(gaze_x, gaze_y)

            # Log data
            self.logger.log_data(pos_x, pos_y)

            print(
                f"Gaze Position (cm): ({gaze_x:.2f}, {gaze_y:.2f}) - {quadrant}, "
                f"Pixels (x,y): ({pos_x}, {pos_y}) - {position}"
            )

            return pos_x, pos_y, position

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

        # start eval of the proccesed calibration
        mean_error, std_error = self.calibration.evaluate_calibration_accuracy()

        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
            while True:
                success, img = webcam.read()
                if not success:
                    print("Error reading from the webcam.")
                    exit(1)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_mp = face_mesh.process(img_rgb)

                if img_mp.multi_face_landmarks:
                    for face_landmarks in img_mp.multi_face_landmarks:
                        face_input, left_eye_input, right_eye_input, face_grid_input = self.extract_features(img, face_landmarks)
                        self.predict_gaze(face_input, left_eye_input, right_eye_input, face_grid_input)

                cv2.imshow("Face Mesh with Gaze Prediction", img)

                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break

        self.logger.save_data()
