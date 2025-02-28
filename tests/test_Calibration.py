import sys
import os
import torch
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tracker.Calibration import CalibrationDataset, Calibration

# Test CalibrationDataset
def test_calibration_dataset():
    sample_data = [
        ((torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224), 
          torch.randn(1, 3, 224, 224), torch.randn(1, 25, 25)), (0.5, -0.5)),
        ((torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224), 
          torch.randn(1, 3, 224, 224), torch.randn(1, 25, 25)), (-0.2, 0.3))
    ]
    dataset = CalibrationDataset(sample_data)
    assert len(dataset) == 2
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 5  # (faces, eyes_left, eyes_right, face_grids, gaze_targets)

# Test Calibration initialization
def test_calibration_init():
    mock_tracker = MagicMock()
    mock_tracker.device = "cpu"
    calibration = Calibration(mock_tracker)
    
    assert calibration.gaze_tracker == mock_tracker
    assert isinstance(calibration.capture_points, list)
    assert calibration.calibration_done is False
    
# Test Calibration evaluation
def test_evaluate_calibration_accuracy():
    mock_tracker = MagicMock()
    mock_tracker.device = "cpu"
    mock_tracker.model = MagicMock()
    mock_tracker.model.eval = MagicMock()
    mock_tracker.model.side_effect = lambda *args: torch.tensor([[0.0, 0.0]])

    calibration = Calibration(mock_tracker)
    calibration.capture_points = [
        ((torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224),
          torch.randn(1, 3, 224, 224), torch.randn(1, 25, 25)), (0.5, -0.5))
    ]
    
    mean_error, std_error = calibration.evaluate_calibration_accuracy()
    assert mean_error >= 0
    assert std_error >= 0