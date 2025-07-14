#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`GazeTrain` module
=========================

This module handles full training of the gaze prediction model using the MPIIFaceGaze dataset.
It processes the data, constructs the training pipeline, and trains the GazeModel.

:author: Pather Stevenson
:date: July 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tracker import GazeModel 
from utils.mpiifacegaze_dataset import *

def GazeTrain(
    root_dir: str,
    model_save_path: str = "../models/itracker_mpiiface_gradient.pth",
    epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> GazeModel:
    """
    Train gaze model with low-memory batch-wise loading of .pkl feature files.

    Each epoch:
    - Loads each train .pkl file in order and trains on it.
    - Then evaluates on all test .pkl files.

    :param root_dir: Directory with 'train/' and 'test/' subfolders of .pkl batches
    :param model_save_path: Path to save the final model
    :param epochs: Number of training epochs
    :param batch_size: Mini-batch size for each loaded .pkl batch
    :param learning_rate: Learning rate for optimizer
    :param device: 'cuda' or 'cpu'
    :return: Trained GazeModel
    """
    train_dir = os.path.join(root_dir, "train/")
    test_dir = os.path.join(root_dir, "test/")
    train_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".pkl")])
    test_files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".pkl")])

    model = GazeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        train_samples = 0

        # ========== TRAIN ==========
        for pkl_file in tqdm(train_files, desc="Training on batches"):
            dataset = FaceGazeBatchDataset(pkl_file)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for batch in dataloader:
                face, eye_left, eye_right, face_grid, gaze = [b.to(device) for b in batch]

                optimizer.zero_grad()
                pred = model(face, eye_left, eye_right, face_grid)
                loss = criterion(pred, gaze)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * face.size(0)
                train_samples += face.size(0)

        avg_train_loss = train_loss / train_samples
        print(f"→ Train Loss: {avg_train_loss:.6f}")

        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for pkl_file in tqdm(test_files, desc="Validating on batches"):
                dataset = FaceGazeBatchDataset(pkl_file)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                for batch in dataloader:
                    face, eye_left, eye_right, face_grid, gaze = [b.to(device) for b in batch]
                    pred = model(face, eye_left, eye_right, face_grid)
                    loss = criterion(pred, gaze)

                    val_loss += loss.item() * face.size(0)
                    val_samples += face.size(0)

        avg_val_loss = val_loss / val_samples
        print(f"→ Val Loss: {avg_val_loss:.6f}")

    # Save final model
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    return model