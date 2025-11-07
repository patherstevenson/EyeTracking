#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`main` module
==================

Main entry point for the gaze tracking system.

This module initializes the webcam, loads the gaze tracking model, and starts the tracking process.

:author: Pather Stevenson
:date: February 2025
"""

import cv2
import os
from tracker.GazeTracker import GazeTracker

def main() -> None:
    """
    Main function to start the gaze tracking system.

    - Retrieves the webcam URL from environment variables.
    - Initializes the webcam (local or network stream).
    - Starts the gaze tracking process.
    """
    # Retrieve the webcam URL from environment variables
    webcam_url: str = os.getenv("WEBCAM_URL", "0")  # Default to "0" for local webcam

    # Use the URL if provided; otherwise, default to the local webcam
    webcam = cv2.VideoCapture(webcam_url if webcam_url != "0" else 0)

    if not webcam.isOpened():
        print("Unable to open webcam. Please check your device or URL.")
        return

    gaze_tracker = GazeTracker(model_path="itracker_mpiiface.tar")

    try:
        gaze_tracker.run(webcam)
    except KeyboardInterrupt:
        print("\nTracking stopped by user.")
    finally:
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
