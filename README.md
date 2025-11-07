# EyeTheia

## Project Overview

**EyeTheia** is an open-source project dedicated to **2D gaze estimation**, predicting the user's **point of regard on the screen** in pixel coordinates.  
It leverages pre-trained deep learning models to infer the gaze position from facial images and landmarks extracted in real time.

A **personal calibration phase** is performed for each user to collect sample data across multiple screen points.  
These samples are then used to **fine-tune the model**, adapting it to the user's specific facial features and improving gaze prediction accuracy.

## Environment Setup

We use **Conda** to manage the development environment.  
To create the environment, run:

```bash
$ conda create -n eyetheia python=3.10
$ conda activate eyetheia
```

To install all dependencies:

```bash
$ make lib
```

## Running EyeTheia

You have two main ways to run **EyeTheia**, depending on your use case:

### 1. Run the full demo (tracking + calibration)

If you want to directly try the complete demo — including the calibration phase and real-time gaze tracking — simply run:

```bash
$ make run
```

This command launches the end-to-end application locally, handling camera input, calibration, and live gaze prediction.

## Start a tracker server (API mode)

If you prefer to use EyeTheia as a backend service via its FastAPI interface, you can start a tracker server manually or through the Makefile.

Supported models :

- **baseline** — iTracker trained on the **GazeCapture** dataset  
- **mpiiface** — iTracker retrained on the **MPIIFaceGaze** dataset

You can start them directly using the **Makefile** commands:

```bash
# Launch the baseline tracker (iTracker trained on GazeCapture)
$ make baseline
```
```bash
# Launch the MPIIFaceGaze retrained tracker
$ make mpii
```

Or manually:

```bash
$ python src/run_server.py --model_path MODEL_PATH [--host HOST]
```

- The --model_path argument is mandatory, specifying which model weights to load.
- The --host argument is optional (default: 127.0.0.1).
- The port is automatically assigned based on the selected model :
  - port 8001 : baseline
  - port 8002 for mpiiface

*Note: Each tracker runs its own FastAPI server, allowing multiple instances to operate simultaneously on different ports.*

This mode exposes an API for external clients (e.g., JavaScript frontends using MediaPipe) to send frames and receive gaze predictions in real time.

## API Usage Example

As a demonstration, we have developed a **JavaScript frontend** that interacts directly with the tracker API.  
It handles webcam capture via **MediaPipe**, sends facial landmarks to the backend, and receives real-time gaze predictions.

- [pygaze.js – Calypso frontend example](https://git.interactions-team.fr/INTERACTIONS/calypso/src/branch/main/src/experiment/trackers/pygaze.js)

This implementation can serve as a reference for integrating **EyeTheia** into web-based experimental setups or **interactive applications** involving gaze-based control or accessibility.

## Documentation

We use **Sphinx** to generate project documentation.  
To build the documentation:

```bash
$ make doc
```

The generated HTML files will be available in the `docs/_build/html/` directory.

The complete presentation of the project will be avaible in the `docs/_build/html/index.html` page.

## Testing

To run unit tests:

```bash
$ make test
```

## Dataset

The project supports two pre-trained model configurations:

- **`itracker_baseline.tar`** — based on the original *iTracker* architecture from the paper  
  [*Eye Tracking for Everyone*](https://arxiv.org/abs/1606.05814), trained on the **GazeCapture** dataset.  
  This model provides a strong baseline for general-purpose 2D gaze estimation.

- **`itracker_mpiiface.tar`** — a version **retrained from scratch** using the  
  [**MPIIFaceGaze**](http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip) dataset, which contains real-world face images with accurate gaze annotations.

If you plan to train the model, download and extract the MPIIFaceGaze dataset into the `dataset/` directory at the root of the project.  
For **inference only**, this dataset is **not required**.

## VM User Webcam

If you are not using the FastAPI routes for inference and are running the project inside a **virtual machine (VM)** such as **WSL2**,  
please refer to the subsection [`Running on a Virtual Machine (e.g., WSL2)`](docs/_build/html/index.html) in the generated documentation.

That section provides detailed instructions on how to stream your webcam using **MJPEG Streamer**,  
and how to configure the environment variable `WEBCAM_URL` to enable webcam access within the project.

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.  
You may redistribute and/or modify it under the terms of the GPL-3.0 as published by the Free Software Foundation.

See the [`LICENSE`](LICENSE) file for full details.