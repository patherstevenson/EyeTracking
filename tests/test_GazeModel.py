import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tracker.GazeModel import FeatureImageModel, FaceImageModel, FaceGridModel, GazeModel

# Test FeatureImageModel
def test_feature_image_model():
    model = FeatureImageModel()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape[0] == 1  # Batch size should remain the same
    assert len(output.shape) == 2  # Output should be a flattened feature vector

# Test FaceImageModel
def test_face_image_model():
    model = FaceImageModel()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (1, 64)  # Expected shape from FC layers

# Test FaceGridModel
def test_face_grid_model():
    model = FaceGridModel()
    input_tensor = torch.randn(1, 25, 25)
    output = model(input_tensor)
    assert output.shape == (1, 128)  # Expected shape from FC layers

# Test GazeModel
def test_gaze_model():
    model = GazeModel()
    faces = torch.randn(1, 3, 224, 224)
    eyes_left = torch.randn(1, 3, 224, 224)
    eyes_right = torch.randn(1, 3, 224, 224)
    face_grids = torch.randn(1, 25, 25)
    output = model(faces, eyes_left, eyes_right, face_grids)
    assert output.shape == (1, 2)  # Gaze output should be (x, y)
