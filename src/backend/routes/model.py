from fastapi import APIRouter, HTTPException
from src.tracker.CalibrationDataset import CalibrationDataset
from src.backend.shared.state import capture_points, gaze_tracker


router = APIRouter()

@router.post("/fine_tune")
async def fine_tune_model():
    """
    Fine-tune le modèle `GazeTracker` avec les données de calibration stockées dans `capture_points`.
    """
    if not capture_points:
        raise HTTPException(status_code=400, detail="No calibration data available for fine-tuning.")

    try:
        gaze_tracker.train(CalibrationDataset(capture_points))

        return {"message": "Model fine-tuned successfully", "total_points": len(capture_points)}

    except Exception as e:
        return {"error": str(e)}
