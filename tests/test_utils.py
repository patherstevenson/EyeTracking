import numpy as np
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils.utils import *

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

### Test euclidan_distance_radius
@pytest.mark.parametrize("pt1, pt2, radius, expected", [
    ((0.0, 0.0), (3.0, 4.0), 5.0, True),   # Distance = 5.0, equal to the radius
    ((0.0, 0.0), (6.0, 8.0), 5.0, False),  # Distance = 10.0, greater than the radius
    ((1.0, 1.0), (4.0, 5.0), 5.0, True),   # Distance = 5.0, equal to the radius
    ((0.0, 0.0), (0.0, 0.0), 1.0, True),   # Same point, distance = 0.0
    ((0.0, 0.0), (0.5, 0.5), 1.0, True),   # Distance = 0.707, less than the radius
])
def test_euclidan_distance_radius(pt1, pt2, radius, expected):
    """
    Test whether the function correctly determines if the Euclidean distance between two points is within a given radius.
    """
    assert euclidan_distance_radius(pt1, pt2, radius) == expected

### Test draw_bounding_boxes
def test_draw_bounding_boxes():
    """
    Test if draw_bounding_boxes correctly draws bounding boxes on the given frame.

    The test:
    - Creates a blank image (480x640)
    - Draws bounding boxes for the face, left eye, and right eye
    - Checks if the pixels in the expected bounding box areas have changed
    """
    # Create a blank black image (480x640)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Define bounding box coordinates (x, y, width, height)
    face_bbox = (100, 100, 150, 200)
    left_eye_bbox = (120, 130, 40, 30)
    right_eye_bbox = (180, 130, 40, 30)

    # Apply the function
    modified_frame = draw_bounding_boxes(frame.copy(), face_bbox, left_eye_bbox, right_eye_bbox)

    # Validate that the pixels in the bounding box areas have changed
    x, y, w, h = face_bbox
    assert np.any(modified_frame[y:y+h, x:x+w] != frame[y:y+h, x:x+w]), "Face bounding box not drawn correctly"

    x, y, w, h = left_eye_bbox
    assert np.any(modified_frame[y:y+h, x:x+w] != frame[y:y+h, x:x+w]), "Left eye bounding box not drawn correctly"

    x, y, w, h = right_eye_bbox
    assert np.any(modified_frame[y:y+h, x:x+w] != frame[y:y+h, x:x+w]), "Right eye bounding box not drawn correctly"