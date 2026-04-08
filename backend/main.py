from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import pipeline, inventory

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.cv_pipeline import get_sam_engine
    # Booting the engine loads the weights into GPU/CPU memory
    get_sam_engine()
    yield

app = FastAPI(title="VisionPhase ML Pipeline API", lifespan=lifespan)

# Setup CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the assets directory (ikea_dataset)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.normpath(os.path.join(_BASE_DIR, "../public/assets"))
app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

app.include_router(pipeline.router, prefix="/api")
app.include_router(inventory.router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "sam_vit_b"}
