# PFE Eye Tracking

2025 - Master 2 Machine Learning - PFE

## Author

- **Stevenson Pather**  
  M2 Machine Learning Student​  
  University of Lille 

## Supervisor

- **Deise Santana Maia, Ph.D.**  
  Assistant Associate Professor, 3DSAM CRIStAL, IUT A​  
  University of Lille  

## Project Overview

This project focuses on **eye tracking** using machine learning techniques, leveraging pre-trained models to predict gaze direction from image data.  
The goal is to improve the accessibility and accuracy of gaze tracking in various applications, including assistive technologies and human-computer interaction.

## Environment Setup

We use **Conda** to manage the development environment.  
To create the environment, run:

```bash
$ conda create -n pfe python=3.12
$ conda activate pfe
```

To install all dependencies:

```bash
$ make lib
```

## Project Structure

```
EyeTracking/
├── LICENSE                     # License for the project
├── Makefile                    # Commands for installation, testing, and documentation
├── README.md                   # Main project documentation (overview, installation, usage)
├── conf.py                     # Sphinx configuration for documentation
├── images/                     # Images used in documentation
├── notebooks/                  # Jupyter Notebooks for visualization and experiments
├── requirements.txt            # List of dependencies for setting up the project
├── sourcedoc/                  # Sphinx-generated documentation sources
├── src/                        # Main source code directory
│   ├── experiments/            # Saved gaze tracking experiment results
│   │   ├── *.pkl               # Pickle files containing gaze tracking data
│   ├── main.py                 # Main script for running gaze tracking
│   ├── mat/                    # Precomputed mean face and eye datasets
│   │   ├── mean_face_224.mat   # Mean face data
│   │   ├── mean_left_224.mat   # Mean left eye data
│   │   └── mean_right_224.mat  # Mean right eye data
│   ├── models/                 # Pre-trained and fine-tuned gaze tracking models
│   │   └── checkpoint.pth.tar  # Saved model checkpoint
│   ├── tracker/                # Core modules for gaze tracking
│   │   ├── Calibration.py      # Calibration process for improving model accuracy
│   │   ├── GazeDataLogger.py   # Handles logging of gaze tracking data
│   │   ├── GazeModel.py        # The deep learning model for gaze prediction
│   │   ├── GazeTracker.py      # Main gaze tracking logic
│   └── utils/                  # Utility functions and configurations
│       ├── calib_13/*.png      # Calibration grid (13 points)
│       ├── calib_9/*.png       # Calibration grid (9 points)
│       ├── calib_5/*.png       # Calibration grid (5 points)
│       ├── config.py           # Configuration parameters (screen size, calibration settings)
│       ├── utils.py            # Helper functions for preprocessing and gaze tracking
├── tests/                      # Unit tests for the project
│   ├── test_Calibration.py     # Tests for the Calibration module
│   ├── test_GazeDataLogger.py  # Tests for GazeDataLogger
│   ├── test_GazeModel.py       # Tests for GazeModel
│   ├── test_GazeTracker.py     # Tests for GazeTracker
│   ├── test_utils.py           # Tests for utility functions
└── docs/                       # Sphinx documentation (to be generated)
```

## Installation & Dependencies

Install project dependencies:

```bash
$ make lib
```

## VM User Webcam

If you are using a **virtual machine (VM)** such as **WSL2**, please refer to the subsection  
[`Running on a Virtual Machine (e.g., WSL2)`](docs/_build/html/index.html) in the generated documentation.

This section provides detailed instructions on how to stream your webcam using **MJPEG Streamer**  
and configure the environment variable `WEBCAM_URL` to enable webcam support in the project.

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

## License

This project is under the **MIT License**. See the `LICENSE` file for details.

## Contact

For any questions or contributions, feel free to contact **Stevenson Pather** or **Deise Santana Maia**
