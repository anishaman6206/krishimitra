from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from os import makedirs

from app.schemas import AskRequest, AskResponse
from app.services.pipeline import answer
from app.config import settings
from app.http import init_http, close_http
from app.utils.cache import init_cache

# Single FastAPI instance
app = FastAPI()

# Single startup event
@app.on_event("startup")
async def startup_event():
    """Initialize cache and HTTP client on startup."""
    await init_cache()
    await init_http()
    print("Cache initialized")
    print("HTTP client initialized")

# Single shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Close HTTP client on shutdown."""
    await close_http()
    print("HTTP client closed")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (NDVI quicklooks)
makedirs(settings.STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# API endpoints
@app.get("/")
async def root():
    return {"ok": True, "service": "KrishiMitra AI", "version": app.version}

@app.get("/health")
async def health():
    return {
        "ok": True,
        "rag_topk": settings.RAG_TOPK,
        "default_crop": settings.DEFAULT_CROP,
        "wx_days": settings.WX_FORECAST_DAYS,
        "ndvi": {
            "aoi_km": settings.SENTINEL_AOI_KM,
            "recent_days": settings.SENTINEL_RECENT_DAYS,
            "prev_days": settings.SENTINEL_PREV_DAYS,
            "gap_days": settings.SENTINEL_GAP_DAYS,
        },
    }

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Fan-out: RAG (+ citations) + tools (price, sell/wait, weather, NDVI) in parallel.
    Returns fused answer + sources + raw tool outputs for transparency.
    """
    return await answer(req)