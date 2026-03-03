from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import pipeline

app = FastAPI(title="VisionPhase ML Pipeline API")

# Setup CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "sam_vit_b"}

# Pre-load model on startup
@app.on_event("startup")
async def startup_event():
    from app.core.cv_pipeline import get_sam_engine
    # Booting the engine loads the weights into MPS memory
    get_sam_engine()
