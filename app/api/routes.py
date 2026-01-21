"""
API Routes - Image Analysis Endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
from datetime import datetime

from app.services.hash_service import HashService
from app.services.metadata_service import MetadataService
from app.services.detection_service import DetectionService
from app.services.pipeline_service import PipelineService
from app.models.schemas import (
    AnalysisResult,
    BatchAnalysisResult,
)

router = APIRouter()

# Service Instances
hash_service = HashService()
metadata_service = MetadataService()
detection_service = DetectionService()
pipeline_service = PipelineService()


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_single_image(file: UploadFile = File(...)):
    """
    Single Image Analysis

    Executes 3 layers sequentially and returns a comprehensive verdict.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    try:
        contents = await file.read()
        result = await pipeline_service.analyze_image(
            image_bytes=contents, filename=file.filename
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch", response_model=BatchAnalysisResult)
async def analyze_batch_images(
    files: List[UploadFile] = File(...),
):
    """
    Batch Image Analysis (Max 50)
    """
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Up to 50 files allowed.")

    results = []
    for file in files:
        if file.content_type and file.content_type.startswith("image/"):
            try:
                contents = await file.read()
                result = await pipeline_service.analyze_image(
                    image_bytes=contents, filename=file.filename
                )
                results.append(result)
            except Exception as e:
                results.append(
                    {"filename": file.filename, "error": str(e), "status": "failed"}
                )

    # Calculate Statistics
    total = len(results)
    ai_detected = sum(
        1
        for r in results
        if isinstance(r, dict) and r.get("final_verdict") == "ai_generated"
    )
    real_detected = sum(
        1
        for r in results
        if isinstance(r, dict) and r.get("final_verdict") == "likely_real"
    )

    return BatchAnalysisResult(
        total_processed=total,
        ai_generated_count=ai_detected,
        likely_real_count=real_detected,
        uncertain_count=total - ai_detected - real_detected,
        results=results,
    )
