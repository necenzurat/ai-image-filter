"""
Pydantic Schemas - 요청/응답 모델 정의
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class VerdictType(str, Enum):
    AI_GENERATED = "ai_generated"
    LIKELY_REAL = "likely_real"
    UNCERTAIN = "uncertain"


class HashResult(BaseModel):
    is_ai: bool = False
    similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="DinoV2 벡터 유사도")


class MetadataResult(BaseModel):
    model_config = {"extra": "ignore"}  # 추가 필드 무시

    has_c2pa: bool = False
    c2pa_info: Optional[Dict[str, Any]] = None
    exif_data: Optional[Dict[str, Any]] = None
    ai_tool_signatures: List[str] = []
    software_used: Optional[str] = None
    creation_date: Optional[str] = None
    exif_authenticity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="EXIF 진위성 점수")
    exif_inconsistencies: List[str] = Field(default_factory=list, description="EXIF 비정상 패턴 목록")


class DetectionResult(BaseModel):
    model_name: str
    is_ai_generated: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_scores: Optional[Dict[str, float]] = None


class LayerResult(BaseModel):
    layer_name: str
    passed: bool
    details: Dict[str, Any]
    execution_time_ms: float


class AnalysisResult(BaseModel):
    id: str
    filename: str
    analyzed_at: datetime
    
    # 각 Layer 결과
    hash_result: HashResult
    metadata_result: MetadataResult
    detection_result: Optional[DetectionResult] = None
    
    # 종합 판정
    final_verdict: VerdictType
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    
    # 메타 정보
    total_execution_time_ms: float
    layers_executed: List[str]


class BatchAnalysisResult(BaseModel):
    total_processed: int
    ai_generated_count: int
    likely_real_count: int
    uncertain_count: int
    results: List[Any]
    processing_time_seconds: Optional[float] = None


class ImageRecord(BaseModel):
    id: str
    filename: str
    verdict: VerdictType
    confidence: float
    similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="DinoV2 벡터 유사도")
    analyzed_at: datetime
    metadata: Optional[Dict[str, Any]] = None
