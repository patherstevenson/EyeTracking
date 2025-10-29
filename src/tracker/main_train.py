#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main training entrypoint for MPIIFaceGaze.
Supports both PKL-batch mode (low-memory) and full image-streaming mode (HPC).
"""

import argparse
import torch
import os
from tracker.GazeTrain import GazeTrain


def parse_args():
    parser = argparse.ArgumentParser(description="Train gaze model (MPIIFaceGaze)")

    # Data mode
    parser.add_argument("--data-mode", choices=["pkl", "img"], default="pkl",
                        help="Choose between 'pkl' (pre-batched features) or 'img' (direct from dataset).")

    parser.add_argument("--pkl-root", default="../dataset/MPIIFaceGaze/batches",
                        help="Root folder containing train/test PKL batches.")
    parser.add_argument("--img-root", default="../dataset/MPIIFaceGaze",
                        help="Root folder containing raw MPIIFaceGaze participant folders.")

    # Training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision training.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format (may boost performance).")

    # DataLoader
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")

    # Checkpoints
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Path to store checkpoints.")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from (or 'auto').")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"- Device: {device}")
    print(f"- Mode: {args.data_mode}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model = GazeTrain(
        data_mode=args.data_mode,
        pkl_root=args.pkl_root,
        img_root=args.img_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_accum=args.grad_accum,
        amp=args.amp,
        channels_last=args.channels_last,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        resume=args.resume,
        device=device,
    )
