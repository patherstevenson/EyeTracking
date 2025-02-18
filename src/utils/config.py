#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`config` module
====================

Configuration module for screen and gaze tracking settings.

This module defines constants for screen dimensions and gaze prediction range.

:author: Pather Stevenson
:date: February 2025
"""

# Screen dimensions in pixels
SCREEN_WIDTH: int = 2560
SCREEN_HEIGHT: int = 1440
MID_X: int = SCREEN_WIDTH // 2
MID_Y: int = SCREEN_HEIGHT // 2

# Gaze prediction range in cm
GAZE_RANGE_CM: int = 25
