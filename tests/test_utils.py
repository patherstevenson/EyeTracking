import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils.utils import (
    loadMetadata, get_bounding_box, preprocess_roi, 
    generate_face_grid, gaze_cm_to_pixels, pixels_to_gaze_cm
)

from utils.config import SCREEN_WIDTH, SCREEN_HEIGHT


# Test loadMetadata
def test_loadMetadata():
    valid_file = "src/mat/mean_face_224.mat"
    invalid_file = "tests/invalid.mat"
    
    assert isinstance(loadMetadata(valid_file), dict)
    assert loadMetadata(invalid_file) is None

# Test get_bounding_box
def test_get_bounding_box():
    landmarks = [(50, 50), (100, 100), (150, 150)]
    indices = [0, 2]
    bbox = get_bounding_box(indices, landmarks, 10, 10)
    assert bbox == (40, 40, 160, 160)

# Test preprocess_roi
def test_preprocess_roi():
    roi = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    output = preprocess_roi(roi)
    assert output.shape == (1, 224, 224)
    assert output.min() >= 0 and output.max() <= 1

# Test generate_face_grid
def test_generate_face_grid():
    face_bbox = (10, 10, 100, 100)
    image_shape = (200, 200, 3)
    face_grid = generate_face_grid(face_bbox, image_shape)
    assert face_grid.shape == (25 * 25,)
    assert np.sum(face_grid) > 0

# Test gaze_cm_to_pixels
def test_gaze_cm_to_pixels():
    x_pixel, y_pixel = gaze_cm_to_pixels(0, 0)
    assert x_pixel == SCREEN_WIDTH // 2
    assert y_pixel == SCREEN_HEIGHT // 2

# Test pixels_to_gaze_cm
def test_pixels_to_gaze_cm():
    x_cm, y_cm = pixels_to_gaze_cm(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    assert abs(x_cm) < 1e-5
    assert abs(y_cm) < 1e-5
