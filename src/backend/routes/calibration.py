from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from src.utils.utils import pixels_to_gaze_cm

router = APIRouter()

calibration_data = []

class CalibrationPoint(BaseModel):
    x: int
    y: int

@router.post("/submit_calibration")
async def submit_calibration(points: List[CalibrationPoint]):
    """
    Receives calibration points from the web interface.
    Converts pixel coordinates to gaze coordinates in cm.
    """
    global calibration_data
    calibration_data = [
        pixels_to_gaze_cm(point.x, point.y) for point in points
    ]
    return {"message": "Calibration data received", "points": calibration_data}
