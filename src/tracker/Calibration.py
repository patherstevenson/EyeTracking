#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`Calibration` module
=========================

This module handles the calibration process for fine-tuning the gaze tracking model.
It captures gaze targets using mouse clicks and extracts corresponding features for fine-tuning.

:author: Pather Stevenson
:date: February 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import pixels_to_gaze_cm, get_numbered_calibration_points, euclidan_distance_radius
from utils.config import SCREEN_WIDTH, SCREEN_HEIGHT, CALIBRATION_IMAGE_PATH, CALIBRATION_PTS
from tracker.CalibrationDataset import CalibrationDataset

class Calibration:
    """
    Handles the calibration process for fine-tuning the gaze tracking model.
    Uses mouse clicks to capture real gaze targets and extract corresponding features.
    """

    def __init__(self, gaze_tracker: "GazeTracker") -> None:
        """
        Initializes the Calibration object.

        :param gaze_tracker: Instance of GazeTracker to extract gaze features.
        :type gaze_tracker: GazeTracker
        """
        self.gaze_tracker = gaze_tracker
        self.capture_points: list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[float, float]]] = []
        self.margin: int = 20  # Bounding box margin
        self.current_target: tuple[int, int] | None = None  # Latest click position
        self.window_name: str = "Calibration Window"
        self.calibration_done: bool = False
        self.calibration_points = get_numbered_calibration_points()
        self.current_index = 0

        # Load and resize the calibration image
        self.calibration_image = cv2.imread(CALIBRATION_IMAGE_PATH)
        if self.calibration_image is None:
            print(f"Error: Could not load calibration image from {CALIBRATION_IMAGE_PATH}")
            exit(1)

        # Resize the calibration image to fit the screen
        self.calibration_image = cv2.resize(self.calibration_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """
        Mouse click callback function to capture gaze target in a specific order.
        The user must click on the calibration points sequentially from 0 to 12.

        :param event: Type of mouse event.
        :type event: int
        :param x: X coordinate of mouse click.
        :type x: int
        :param y: Y coordinate of mouse click.
        :type y: int
        :param flags: Additional event parameters.
        :type flags: int
        :param param: Extra parameters (unused).
        :type param: any
        """
        if event == cv2.EVENT_LBUTTONDOWN and not self.calibration_done:

            # The next expected point
            expected_point = self.calibration_points[self.current_index]

            if euclidan_distance_radius((x, y), expected_point, SCREEN_WIDTH // 32):
                print(f"Correct click at point {self.current_index} ({x}, {y})")

                self.current_target = (x, y)

                # Load the next calibration image
                self.current_index += 1
                if self.current_index < len(self.calibration_points):
                    new_image_path = CALIBRATION_IMAGE_PATH.rstrip("0.png") + f"{self.current_index}.png"
                    self.calibration_image = cv2.imread(new_image_path)

                    if self.calibration_image is None:
                        print(f"Error: Could not load {new_image_path}")
                        exit(1)
                    self.calibration_image = cv2.resize(self.calibration_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
                else:
                    self.calibration_done = True
            else:
                print(f"Incorrect click at ({x}, {y}). Please click on point {self.current_index}!")

    def evaluate_calibration_accuracy(self) -> tuple[float, float]:
        """
        Evaluates the accuracy of the fine-tuned gaze tracking model on calibration points.

        :return: Mean Euclidean distance error and standard deviation in cm.
        :rtype: tuple[float, float]
        """
        if not self.capture_points:
            print("No calibration data available for evaluation.")
            return 0.0, 0.0

        print("\nEvaluation of calibration started")

        total_errors = []

        self.gaze_tracker.model.eval()

        with torch.no_grad():
            for (face_input, left_eye_input, right_eye_input, face_grid_input), (gaze_x_true, gaze_y_true) in self.capture_points:

                # Move tensors to device
                face_input = face_input.to(self.gaze_tracker.device)
                left_eye_input = left_eye_input.to(self.gaze_tracker.device)
                right_eye_input = right_eye_input.to(self.gaze_tracker.device)
                face_grid_input = face_grid_input.to(self.gaze_tracker.device)

                # Model prediction
                gaze_prediction = self.gaze_tracker.model(face_input, left_eye_input, right_eye_input, face_grid_input)
                gaze_x_pred, gaze_y_pred = gaze_prediction.cpu().numpy().flatten()


                # Compute Euclidean error in cm
                error =  np.linalg.norm(np.array([gaze_x_pred, gaze_y_pred]) - np.array([gaze_x_true, gaze_y_true]))
                total_errors.append(error)

                print(f"True Gaze: ({gaze_x_true:.2f}, {gaze_y_true:.2f}), "
                      f"Predicted Gaze: ({gaze_x_pred:.2f}, {gaze_y_pred:.2f}), "
                      f"Error: {error:.2f}")

        # Compute mean and standard deviation
        mean_error = np.mean(total_errors)
        std_error = np.std(total_errors)

        print(f"\nCalibration Accuracy: Mean Error = {mean_error:.2f}, Std Dev = {std_error:.2f}\n")

        return mean_error, std_error

    def run_calibration(self, webcam: cv2.VideoCapture) -> CalibrationDataset:
        """
        Runs the calibration process using mouse clicks to capture gaze targets.
        The user freely clicks anywhere on the screen to provide gaze samples.

        :param webcam: OpenCV VideoCapture object.
        :type webcam: cv2.VideoCapture
        :return: A CalibrationDataset object containing collected samples.
        :rtype: CalibrationDataset
        """
        print("\nCalibration started")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
            while not self.calibration_done:
            
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow(self.window_name, self.calibration_image)
                cv2.waitKey(1)

                success, img = webcam.read()
                if not success:
                    print("Error reading from the webcam.")
                    exit(1)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_mp = face_mesh.process(img_rgb)

                if img_mp.multi_face_landmarks:
                    for face_landmarks in img_mp.multi_face_landmarks:
                        face_input, left_eye_input, right_eye_input, face_grid_input = self.gaze_tracker.extract_features(img, face_landmarks, SCREEN_WIDTH, SCREEN_HEIGHT)

                        if self.current_target:
                            user_x, user_y = self.current_target
                            gaze_x, gaze_y = pixels_to_gaze_cm(user_x, user_y)

                            self.capture_points.append(((face_input, left_eye_input, right_eye_input, face_grid_input), (gaze_x, gaze_y)))
                            print(f"Captured: Screen ({user_x}, {user_y}) → Gaze ({gaze_x:.2f}, {gaze_y:.2f})")

                            self.current_target = None

                if len(self.capture_points) >= CALIBRATION_PTS:
                    self.calibration_done = True

        cv2.destroyWindow(self.window_name)
        print("\nCalibration completed.")

        return CalibrationDataset(self.capture_points)
