#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.io
import pandas as pd

from tracker.GazeModel import GazeModel
from utils.mpiifacegaze_dataset import *
from utils.utils import get_groupwise_train_test_split
from torch.amp import autocast, GradScaler


def GazeTrain(
    data_mode: str = "pkl",
    pkl_root: str = "../dataset/MPIIFaceGaze/batches",
    img_root: str = "../dataset/MPIIFaceGaze",
    epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    grad_accum: int = 1,
    amp: bool = False,
    channels_last: bool = False,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    checkpoint_dir: str = "checkpoints",
    save_every: int = 1,
    resume: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    smooth_l1_delta: float = 1.0
):
    """
    Universal gaze training function for both local (low-memory) and HPC cluster use.
    Supports PKL-based batch streaming and full image mode.

    :param data_mode: "pkl" or "img"
    :param pkl_root: path to PKL batches (train/test folders)
    :param img_root: path to MPIIFaceGaze dataset
    :param epochs: number of epochs
    :param batch_size: mini-batch size
    :param learning_rate: learning rate
    :param grad_accum: gradient accumulation steps
    :param amp: whether to use mixed precision
    :param channels_last: enable channels_last memory format
    :param num_workers: dataloader workers
    :param prefetch_factor: prefetch factor for dataloader
    :param pin_memory: pin memory for CUDA transfer
    :param persistent_workers: keep dataloader workers alive
    :param checkpoint_dir: directory to store checkpoints
    :param save_every: save checkpoint every N epochs
    :param resume: "auto" or checkpoint path
    :param device: "cuda" or "cpu"
    :param smooth_l1_delta: delta threshold for SmoothL1Loss
    """

    # --- Model and device setup ---
    model = GazeModel().to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss(beta=smooth_l1_delta)
    scaler = GradScaler(device="cuda", enabled=amp)

    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 0

    # --- Save run configuration ---
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "w") as cfg:
        json.dump({
            "data_mode": data_mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "grad_accum": grad_accum,
            "amp": amp,
            "channels_last": channels_last,
            "smooth_l1_delta": smooth_l1_delta,
            "device": str(device)
        }, cfg, indent=2)

    # --- Resume from checkpoint ---
    if resume:
        if resume == "auto":
            candidates = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
            if candidates:
                resume = os.path.join(checkpoint_dir, candidates[-1])

        if resume and os.path.exists(resume):
            print(f"Resuming from checkpoint: {resume}")
            checkpoint = torch.load(resume, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            model.to(device)
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}")

    print(f"Starting training (mode={data_mode}, from epoch {start_epoch})")

    # --- Dataset preparation (train and validation) ---
    train_datasets = []
    val_dataset = None

    if data_mode == "pkl":
        train_dir = os.path.join(pkl_root, "train")
        test_dir = os.path.join(pkl_root, "test")

        train_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".pkl")])
        test_files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".pkl")])

        train_datasets = [FaceGazeBatchDataset(f) for f in train_files]
        val_files = test_files

    else:  # Full image mode (cluster)
        dataset_full = MPIIFaceGazeDataset(img_root)
        df = dataset_full.to_dataframe()
        
        # Load the CSV of skipped images
        skip_csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "skipped_images.csv")

        if os.path.exists(skip_csv_path):
            skipped_df = pd.read_csv(skip_csv_path, delimiter=" ")

            skipped_images = set(skipped_df["img_path"].str.replace("../", "", regex=False))

            print(f"Skipping {len(skipped_images)} problematic images listed in skipped_images.csv")

            # Exclude those images from the DataFrame
            df = df[~df["img_path"].isin(skipped_images)].reset_index(drop=True)
        else:
            print("Warning: skipped_images.csv not found — proceeding with all images.")

    
        df_train, df_test = get_groupwise_train_test_split(df, fold_index=0)

        mat_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "mat"))
        meta_face = scipy.io.loadmat(os.path.join(mat_dir, "mean_face_224_MPIIFace.mat"))
        meta_left = scipy.io.loadmat(os.path.join(mat_dir, "mean_left_224_MPIIFace.mat"))
        meta_right = scipy.io.loadmat(os.path.join(mat_dir, "mean_right_224_MPIIFace.mat"))
        means = {
            "face": torch.tensor(meta_face["mean_face"], dtype=torch.float32),
            "eye_left": torch.tensor(meta_left["mean_eye_left"], dtype=torch.float32),
            "eye_right": torch.tensor(meta_right["mean_eye_right"], dtype=torch.float32),
        }

        train_datasets = [FaceGazeDataset(df_train, means=means, face_mesh=None)]
        val_dataset = FaceGazeDataset(df_test, means=means, face_mesh=None)

    # --- Training loop ---
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for dataset in train_datasets:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
                faces, eye_left, eye_right, face_grid, gaze = [b.to(device) for b in batch]
                if channels_last:
                    faces = faces.contiguous(memory_format=torch.channels_last)
                    eye_left = eye_left.contiguous(memory_format=torch.channels_last)
                    eye_right = eye_right.contiguous(memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", enabled=amp):
                    pred = model(faces, eye_left, eye_right, face_grid)
                    loss = criterion(pred, gaze)

                scaler.scale(loss).backward()

                if (i + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item() * faces.size(0)
                train_samples += faces.size(0)

        avg_train_loss = train_loss / train_samples
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}")

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            if data_mode == "pkl":
                for pkl_file in val_files:
                    dataset = FaceGazeBatchDataset(pkl_file)
                    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=pin_memory)
                    for batch in val_loader:
                        faces, eye_left, eye_right, face_grid, gaze = [b.to(device) for b in batch]
                        pred = model(faces, eye_left, eye_right, face_grid)
                        loss = criterion(pred, gaze)
                        val_loss += loss.item() * faces.size(0)
                        val_samples += faces.size(0)
            else:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                )
                for batch in val_loader:
                    faces, eye_left, eye_right, face_grid, gaze = [b.to(device) for b in batch]
                    pred = model(faces, eye_left, eye_right, face_grid)
                    loss = criterion(pred, gaze)
                    val_loss += loss.item() * faces.size(0)
                    val_samples += faces.size(0)

        avg_val_loss = val_loss / max(1, val_samples)
        print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.6f}")

        # --- Save checkpoint ---
        if (epoch + 1) % save_every == 0:
            save_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}_val{avg_val_loss:.4f}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_path)
            print(f"Checkpoint saved: {save_path}")

    print("Training complete.")
    return model
