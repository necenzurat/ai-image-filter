"""
AI Image Filter Pipeline - FastAPI Backend
Pipeline for filtering AI-generated images from ML training datasets
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Logic executed on app startup/shutdown"""
    # Startup
    print("✅ Service initialized (Stateless)")
    yield
    # Shutdown
    print("👋 Shutting down...")


app = FastAPI(
    title="AI Image Filter Pipeline",
    description="""
    ## ML Training Data Quality Verification Pipeline
    
    Ensures the quality of training datasets by detecting AI-generated images.
    
    ### 3-Layer Verification System
    - **Layer 1**: Hash Check - Image hash calculation (MD5, SHA256, Perceptual Hash)
    - **Layer 2**: Metadata Analysis - C2PA/EXIF analysis and AI tool signature detection
    - **Layer 3**: AI Detection - ML model-based AI generated image detection
    
    *Stateless Service - No Database Used*
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Router
app.include_router(routes.router, prefix="/api/v1", tags=["Image Analysis"])


@app.get("/", tags=["Health"])
async def root():
    return {"message": "AI Image Filter Pipeline API", "docs": "/docs", "health": "ok"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
