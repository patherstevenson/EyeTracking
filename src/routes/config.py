#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`config` module
=====================

This module defines configuration endpoints for the gaze tracking API.
It currently exposes an endpoint to update the screen resolution used for
coordinate normalization and gaze prediction.

:author: Pather Stevenson
:date: October 2025
"""

from fastapi import APIRouter, Form, Depends
from routes.dependency import get_screen

router = APIRouter()

@router.post("/update_screen")
async def update_screen_size(
    width: int = Form(...),
    height: int = Form(...),
    screen=Depends(get_screen),
):
    """
    Update the screen size (width, height) used by gaze normalization.

    :param width: New screen width in pixels.
    :pram height: New screen height in pixels.
    :type width: int
    :type height: int

    
    :return: Confirmation message with updated values.
    :rtype: dict
    """
    # screen is a mutable list or tuple injected from main (default (1920, 1080)).
    # If it is a tuple (immutable), we rebind its contents in place.
    if isinstance(screen, tuple):
        screen = list(screen)  # convert tuple to mutable list for updates

    screen[0] = width
    screen[1] = height

    return {"message": "Screen size updated", "width": width, "height": height}
