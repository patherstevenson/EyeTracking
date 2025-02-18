#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`GazeDataLogger` module
============================

This module defines the `GazeDataLogger` class, responsible for recording and saving gaze tracking data. 
The gaze positions are stored as `(x, y)` tuples and saved in a timestamped pickle file.

:author: Pather Stevenson
:date: February 2025
"""

import os
import pickle
import datetime
from typing import List, Tuple

class GazeDataLogger:
    """
    A class responsible for logging and saving gaze tracking data.
    The data is recorded as `(x, y)` coordinates and stored in a pickle file.
    """

    def __init__(self) -> None:
        """
        Initializes the GazeDataLogger, setting up the storage directory and 
        defining a unique filename based on the current timestamp.
        """
        self.tracking_data: List[Tuple[int, int]] = []
        self.tracking_folder: str = "src/experiments/"
        os.makedirs(self.tracking_folder, exist_ok=True)

        # Generate a unique filename using the current date and time
        self.timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.tracking_file: str = os.path.join(self.tracking_folder, f"gaze_{self.timestamp}.pkl")

    def log_data(self, pos_x: int, pos_y: int) -> None:
        """
        Logs a new gaze position into the tracking data list.

        :param pos_x: X-coordinate of the gaze position.
        :param pos_y: Y-coordinate of the gaze position.
        """
        self.tracking_data.append((pos_x, pos_y))

    def save_data(self) -> None:
        """
        Saves the recorded gaze tracking data into a pickle file.

        The file is stored in the `src/experiments/` directory with a timestamped filename.
        """
        with open(self.tracking_file, "wb") as f:
            pickle.dump(self.tracking_data, f)
        print(f"Tracking data saved to {self.tracking_file}")
