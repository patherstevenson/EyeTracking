import cv2
import mediapipe as mp
import time

import torch
from torch.utils.data import DataLoader, Dataset

from utils.utils import pixels_to_gaze_cm
from utils.config import SCREEN_WIDTH, SCREEN_HEIGHT, CALIBRATION_IMAGE_PATH, CALIBRATION_PTS


class CalibrationDataset(Dataset):
    """
    PyTorch Dataset for calibration data.
    Used to fine-tune the gaze model with user-specific data.
    """

    def __init__(self, calibration_data):
        """
        Initializes the dataset.

        :param calibration_data: List of tuples (features, gaze_targets)
        :type calibration_data: list
        """
        self.data = calibration_data

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: Number of calibration samples.
        :rtype: int
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        :param idx: Index of the sample.
        :type idx: int
        :return: Tuple (faces, eyes_left, eyes_right, face_grids, gaze_targets).
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        features, gaze_targets = self.data[idx]
        face_input, left_eye_input, right_eye_input, face_grid_input = features

        return (
            face_input.squeeze(0),  # Remove batch dimension
            left_eye_input.squeeze(0),
            right_eye_input.squeeze(0),
            face_grid_input.squeeze(0),
            torch.tensor(gaze_targets, dtype=torch.float32)
        )


class Calibration:
    """
    Handles the calibration process for fine-tuning the gaze tracking model.
    Uses mouse clicks to capture real gaze targets and extract corresponding features.
    """

    def __init__(self, gaze_tracker):
        """
        Initializes the Calibration object.

        :param webcam: OpenCV VideoCapture object.
        :param gaze_tracker: Instance of GazeTracker.
        """
        self.gaze_tracker = gaze_tracker
        self.capture_points = []  # Stores calibration data (features, gaze_x, gaze_y)
        self.margin = 20  # Bounding box margin
        self.current_target = None  # Latest click position
        self.window_name = "Calibration Window"
        self.calibration_done = False
        self.calibration_image = cv2.imread(CALIBRATION_IMAGE_PATH)  # Load calibration image
        self.calibration_image = cv2.resize(self.calibration_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

        if self.calibration_image is None:
            print(f"Error: Could not load calibration image from {CALIBRATION_IMAGE_PATH}")
            exit(1)

        # Resize the calibration image to fit the screen
        self.calibration_image = cv2.resize(self.calibration_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Mouse click callback function to capture gaze target.
        Converts screen pixel coordinates (x, y) to gaze coordinates in cm.

        :param event: Type of mouse event.
        :param x: X coordinate of mouse click.
        :param y: Y coordinate of mouse click.
        :param flags: Additional event parameters.
        :param param: Extra parameters.
        """
        if event == cv2.EVENT_LBUTTONDOWN and not self.calibration_done:
            print(f"Click registered at ({x}, {y})")
            self.current_target = (x, y)  # Store clicked position

    def run_calibration(self, webcam):
        """
        Runs the calibration process using mouse clicks to capture gaze targets.
        The user freely clicks anywhere on the screen to provide gaze samples.

        :param webcam: OpenCV VideoCapture object.
        :type webcam: cv2.VideoCapture
        :return: Calibration dataset.
        :rtype: CalibrationDataset
        """
        print("\nCalibration started. Click anywhere on the screen to provide calibration points.")

        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
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
                        face_input, left_eye_input, right_eye_input, face_grid_input = self.gaze_tracker.extract_features(img, face_landmarks)

                        if self.current_target:
                            user_x, user_y = self.current_target
                            gaze_x, gaze_y = pixels_to_gaze_cm(user_x, user_y)

                            self.capture_points.append(((face_input, left_eye_input, right_eye_input, face_grid_input), (gaze_x, gaze_y)))
                            print(f"Captured: Screen ({user_x}, {user_y}) → Gaze ({gaze_x:.2f}, {gaze_y:.2f}) cm")

                            self.current_target = None

                if len(self.capture_points) >= CALIBRATION_PTS:
                    self.calibration_done = True

        cv2.destroyWindow(self.window_name)
        print("Calibration completed.")

        return CalibrationDataset(self.capture_points)
