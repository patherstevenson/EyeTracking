from fastapi import APIRouter, Form
from src.backend.shared import state

router = APIRouter()

@router.post("/update_screen")
async def update_screen_size(
    width: int = Form(...),
    height: int = Form(...)
):
    state.screen_width = width
    state.screen_height = height
    return {"message": "Screen size updated", "width": width, "height": height}
