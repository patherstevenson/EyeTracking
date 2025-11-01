import torch
from unittest.mock import MagicMock, patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tracker.GazeTracker import GazeTracker
from tracker.GazeModel import GazeModel
from tracker.GazeDataLogger import GazeDataLogger

# Test initialization
def test_gaze_tracker_init():
    with patch("tracker.GazeTracker._load_model", return_value=None), \
         patch("tracker.GazeDataLogger", autospec=True) as mock_gaze_logger:
        
        mock_gaze_logger.return_value = MagicMock()
        tracker = GazeTracker()
    
    assert isinstance(tracker.model, GazeModel)
    assert isinstance(tracker.logger, GazeDataLogger)
    assert str(tracker.device) in ["cpu", "cuda"]

# Test _determine_position and _determine_quadrant
def test_gaze_tracker_determine_position():
    tracker = GazeTracker()
    assert tracker._determine_position(100, 100) in ["Top Left", "Top Right", "Bottom Left", "Bottom Right", "Center"]

def test_gaze_tracker_determine_quadrant():
    tracker = GazeTracker()
    assert tracker._determine_quadrant(-0.5, 0.5) == "Top Left"
    assert tracker._determine_quadrant(0.5, 0.5) == "Top Right"
    assert tracker._determine_quadrant(-0.5, -0.5) == "Bottom Left"
    assert tracker._determine_quadrant(0.5, -0.5) == "Bottom Right"
    assert tracker._determine_quadrant(0.0, 0.0) == "Center"

# Test gaze prediction
def test_gaze_tracker_predict_gaze():
    tracker = GazeTracker()
    tracker.model = MagicMock()
    tracker.logger = MagicMock()
    
    face_input = torch.randn(1, 3, 224, 224)
    left_eye_input = torch.randn(1, 3, 224, 224)
    right_eye_input = torch.randn(1, 3, 224, 224)
    face_grid_input = torch.randn(1, 25, 25)
    
    tracker.model.return_value = torch.tensor([[0.0, 0.0]])
    
    pos_x, pos_y, position = tracker.predict_gaze(face_input, left_eye_input, right_eye_input, face_grid_input)
    
    assert isinstance(pos_x, int)
    assert isinstance(pos_y, int)
    assert isinstance(position, str)

# Test extract_features
def test_gaze_tracker_extract_features():
    tracker = GazeTracker()
    tracker.extract_features = MagicMock(return_value=(torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224), torch.randn(1, 25, 25)))
    
    img_mock = torch.randn(224, 224, 3)  # Fake image
    face_landmarks_mock = MagicMock()
    face_input, left_eye_input, right_eye_input, face_grid_input = tracker.extract_features(img_mock, face_landmarks_mock)
    
    assert face_input.shape == (1, 3, 224, 224)
    assert left_eye_input.shape == (1, 3, 224, 224)
    assert right_eye_input.shape == (1, 3, 224, 224)
    assert face_grid_input.shape == (1, 25, 25)

# Test train
def test_gaze_tracker_train():
    tracker = GazeTracker()
    tracker.train = MagicMock()
    dataset_mock = MagicMock()
    tracker.train(dataset_mock, epochs=1, learning_rate=0.001, batch_size=2)
    tracker.train.assert_called_once()

# Test save tracking data
def test_gaze_tracker_save_tracking_data():
    tracker = GazeTracker()
    tracker.logger = MagicMock()
    
    tracker.save_tracking_data()
    tracker.logger.save_data.assert_called_once()
