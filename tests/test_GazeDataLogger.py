import os
import pickle
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tracker.GazeDataLogger import GazeDataLogger

def test_gaze_data_logger_init():
    logger = GazeDataLogger()
    assert isinstance(logger.tracking_data, list)
    assert len(logger.tracking_data) == 0
    assert os.path.exists(logger.tracking_folder)
    assert logger.tracking_file.startswith(logger.tracking_folder)

def test_gaze_data_logger_log_data():
    logger = GazeDataLogger()
    logger.log_data(100, 200)
    logger.log_data(300, 400)
    assert len(logger.tracking_data) == 2
    assert logger.tracking_data[0] == (100, 200)
    assert logger.tracking_data[1] == (300, 400)

def test_gaze_data_logger_save_data(tmp_path):
    logger = GazeDataLogger()
    logger.tracking_folder = str(tmp_path)  # Use a temporary directory
    logger.tracking_file = os.path.join(logger.tracking_folder, "gaze_test.pkl")
    
    logger.log_data(100, 200)
    logger.log_data(300, 400)
    logger.save_data()
    
    assert os.path.exists(logger.tracking_file)
    
    with open(logger.tracking_file, "rb") as f:
        loaded_data = pickle.load(f)
    
    assert loaded_data == [(100, 200), (300, 400)]
