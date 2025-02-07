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
PFE-Eye-Tracking/
│── notebooks/           # Jupyter notebooks
│── src/                 # Source code
│   ├── models/          # ...
│   ├── dot_tracking/    # ...
│   ├── mat/             # ...
│── tests/               # Unit tests
│── docs/                # Documentation (Sphinx)
│── README.md            # Project documentation
│── requirements.txt     # Python dependencies
│── Makefile             # Commands for installation, testing, and docs
```

## Installation & Dependencies

Install project dependencies:

```bash
$ make lib
```

## Model

We use the pre-trained **GazeCapture** model for eye tracking, based on deep learning techniques.  
The original model was developed by MIT CSAIL and is presented in the paper:  

> **Eye Tracking for Everyone**  
> Krafka et al. (2016)  
> [📄 Read on arXiv](https://arxiv.org/abs/1606.05814)

The model is available at: [GazeCapture GitHub](https://github.com/CSAILVision/GazeCapture).

## Documentation

We use **Sphinx** to generate project documentation.  
To build the documentation:

```bash
$ make doc
```

The generated HTML files will be available in the `docs/_build/html/` directory.

## Testing

To run unit tests:

```bash
$ make test
```

## License

This project is under the **MIT License**. See the `LICENSE` file for details.

## Contact

For any questions or contributions, feel free to contact **Stevenson Pather** or **Deise Santana Maia**
