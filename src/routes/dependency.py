#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`dependency` module
========================

This module declares symbolic dependency providers used across the API routes.
Each function is intended to be overridden per FastAPI application instance
(e.g., in the `lifespan` setup) using `app.dependency_overrides[...]`.

By default, these dependencies raise 500 errors to make improper wiring
immediately visible during development and testing.

Usage pattern (in app bootstrap):
---------------------------------
    app.dependency_overrides[get_tracker] = lambda: tracker_instance
    app.dependency_overrides[get_capture_points] = lambda: capture_points_list
    app.dependency_overrides[get_screen] = lambda: (width, height)

:author: Pather Stevenson
:date: October 2025
"""

from typing import Tuple, List, Any
from fastapi import HTTPException


def get_tracker() -> Any:
    """Return the current GazeTracker instance (to be injected by the app)."""
    raise HTTPException(status_code=500, detail="Tracker not injected")


def get_capture_points() -> List[Any]:
    """
    Return the mutable list that accumulates calibration samples for the current app.
    Each element is typically a tuple: ((face, left_eye, right_eye, face_grid), (x_cm, y_cm)).
    """
    raise HTTPException(status_code=500, detail="capture_points not injected")


def get_screen() -> Tuple[int, int]:
    """Return the current screen (width, height) used for decoding/normalization."""
    raise HTTPException(status_code=500, detail="screen (W,H) not injected")
