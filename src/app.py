#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`app` module
=========================

This module exposes a `create_app` factory to build a FastAPI application
parameterized by the `model_path` (and optionally screen size).
It wires dependencies for shared routers (config, calibration, model),
sets CORS, and returns a ready-to-serve app.

:author: Pather Stevenson
:date: October 2025
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import calibration, config, model
from routes.dependency import get_tracker, get_capture_points, get_screen
from tracker.GazeTracker import GazeTracker


def create_app(
    model_path: str,
    screen: Tuple[int, int] = (1920, 1080),
    allow_origins: list[str] | None = None,
) -> FastAPI:
    """
    Build a FastAPI app configured for the given tracker model_path.

    :param model_path: Path or filename for the tracker weights.
    :param screen: Default (width, height) used by endpoints (can be updated via /config if you expose it).
    :param allow_origins: CORS allowed origins; defaults to local dev hosts.
    """
    if allow_origins is None:
        allow_origins = ["http://127.0.0.1:8000", "http://localhost:8000"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tracker = GazeTracker(model_path=model_path)
        capture_points = []
        try:
            # Dependency injection for this specific application instance
            app.dependency_overrides[get_tracker] = lambda: tracker
            app.dependency_overrides[get_capture_points] = lambda: capture_points
            app.dependency_overrides[get_screen] = lambda: list(screen)
            yield
        finally:
            # If in the future explicit GPU VRAM release or cleanup is required,
            # implement a `.close()` method in GazeTracker and call it here.
            if hasattr(tracker, "close"):
                tracker.close()

    app = FastAPI(lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-type", "Authorization"],
    )

    # Routers
    app.include_router(config.router, prefix="/config")
    app.include_router(calibration.router, prefix="/calibration")
    app.include_router(model.router, prefix="/model")

    return app
