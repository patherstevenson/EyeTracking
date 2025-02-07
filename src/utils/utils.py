
import cv2
import numpy as np
from utils.config import SCREEN_WIDTH, SCREEN_HEIGHT, GAZE_RANGE_CM
import scipy.io as sio


# MediaPipe marker id
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
FACE_OVAL = list(range(10, 338))

def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


def get_bounding_box(indices, landmarks, x_margin=0, y_margin=0):
    coords = [landmarks[i] for i in indices]
    x_min = min([pt[0] for pt in coords]) - x_margin
    y_min = min([pt[1] for pt in coords]) - y_margin
    x_max = max([pt[0] for pt in coords]) + x_margin
    y_max = max([pt[1] for pt in coords]) + y_margin
    return max(0, x_min), max(0, y_min), x_max, y_max

def preprocess_roi(roi, size=(224, 224)):
    roi_resized = cv2.resize(roi, size)
    roi_normalized = roi_resized / 255.0
    roi_prepared = np.expand_dims(roi_normalized, axis=0)
    return roi_prepared

def generate_face_grid(face_bbox, image_shape, grid_size=25):
    face_grid = np.zeros((grid_size, grid_size))
    grid_x = int((face_bbox[0] / image_shape[1]) * grid_size)
    grid_y = int((face_bbox[1] / image_shape[0]) * grid_size)
    grid_w = int(((face_bbox[2] - face_bbox[0]) / image_shape[1]) * grid_size)
    grid_h = int(((face_bbox[3] - face_bbox[1]) / image_shape[0]) * grid_size)
    face_grid[grid_y:grid_y + grid_h, grid_x:grid_x + grid_w] = 1
    return face_grid.flatten()

def gaze_cm_to_pixels(gaze_x_cm, gaze_y_cm):
    x_pixel = (gaze_x_cm + GAZE_RANGE_CM) / (2 * GAZE_RANGE_CM) * SCREEN_WIDTH
    y_pixel = (GAZE_RANGE_CM - gaze_y_cm) / (2 * GAZE_RANGE_CM) * SCREEN_HEIGHT
    
    return int(x_pixel), int(y_pixel)
