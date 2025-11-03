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