#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`CalibrationDataset` module
=========================

This module defines the `CalibrationDataset` class, a custom PyTorch dataset 
used during the calibration. It stores and  manages user-specific gaze samples collected
during the calibration process to fine-tune the gaze estimation model.

Each sample includes the extracted facial features (face, left eye, right eye, 
and face grid) along with the corresponding gaze target coordinates.

:author: Pather Stevenson
:date: February 2025
"""
    
import torch
from torch.utils.data import Dataset


class CalibrationDataset(Dataset):
    """
    PyTorch Dataset for storing calibration data.
    This dataset is used to fine-tune the gaze tracking model with user-specific gaze samples.
    """

    def __init__(self, calibration_data: list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[float, float]]]):
        """
        Initializes the dataset.

        :param calibration_data: List of tuples (features, gaze_targets).
                                 Each entry contains extracted features and the corresponding gaze target.
        :type calibration_data: list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[float, float]]]
        """
        self.data = calibration_data

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        :return: Number of calibration samples.
        :rtype: int
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset.

        :param idx: Index of the sample.
        :type idx: int
        :return: Tuple (faces, eyes_left, eyes_right, face_grids, gaze_targets).
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
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
