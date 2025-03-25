from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.backend.routes import calibration_router
import os

app = FastAPI()

# Servir les fichiers statiques (interface web)
static_path = os.path.abspath("src/backend/static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def serve_calibration_page():
    """
    Serve the calibration page when accessing the API root.
    """
    return FileResponse(os.path.join(static_path, "index.html"))

# Ajouter les routes API
app.include_router(calibration_router, prefix="/calibration", tags=["Calibration"])
#app.include_router(model_router, prefix="/model", tags=["Model"])
