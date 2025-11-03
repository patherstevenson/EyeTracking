#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`run_server` module
========================

CLI launcher for the FastAPI app built by `create_app`. It allows passing
`--model-path`, `--port`

Usage:
    python run_server.py --model-path itracker_mpiiface.tar --port 8002

:author: Pather Stevenson
:date: October 2025
"""

import os
import sys
import argparse
import uvicorn

from app import create_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path/filename to tracker weights (e.g., itracker_mpiiface.tar)")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the server")
    args = parser.parse_args()

    port = None

    match args.model_path:
        case "itracker_baseline.tar":
            port = 8001
        case "itracker_mpiiface.tar":
            port = 8002
        case _:
            raise ValueError("Unknown model_path")
            
    app = create_app(model_path=args.model_path)
    uvicorn.run(app, host=args.host, port=port)
