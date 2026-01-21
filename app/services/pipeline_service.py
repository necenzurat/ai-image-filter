"""
Pipeline Service
Integrates 3 Layers to perform comprehensive verdict
"""

import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional

from app.services.hash_service import HashService
from app.services.metadata_service import MetadataService
from app.services.detection_service import DetectionService
from app.models.schemas import (
    AnalysisResult,
    HashResult,
    MetadataResult,
    DetectionResult,
    VerdictType,
)


class PipelineService:
    """3-Layer Analysis Pipeline Service"""

    def __init__(
        self,
        db_vectors_path: str = "./data/ai_dinohashes.npy",
        metadata_path: str = "./data/ai_metadata.csv",
        similarity_threshold: float = 0.85,
    ):
        """
        PipelineService Initialization

        Args:
            db_vectors_path: AI Image Vector File Path
            metadata_path: AI Image Metadata File Path
            similarity_threshold: DinoV2 Similarity Threshold
        """
        self.hash_service = HashService(
            db_vectors_path=db_vectors_path,
            metadata_path=metadata_path,
            threshold=similarity_threshold if similarity_threshold else 0.85,
        )
        self.metadata_service = MetadataService()
        self.detection_service = DetectionService()

        # Verdict Thresholds
        self.CONFIDENCE_THRESHOLD = 0.7
        self.AI_DETECTION_WEIGHT = 0.3
        self.METADATA_WEIGHT = 0.4
        self.HASH_WEIGHT = 0.3

    async def analyze_image(
        self,
        image_bytes: bytes,
        filename: str,
    ) -> AnalysisResult:
        """
        Execute Comprehensive Image Analysis

        Args:
            image_bytes: Image Binary Data
            filename: Filename
        """
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        layers_executed = []

        # ========== Layer 1: Hash Check ==========
        layer1_start = time.time()
        hash_data = self.hash_service.compute_hash(image_bytes)

        hash_result = HashResult(
            is_ai=hash_data["is_ai"],
            similarity=hash_data["similarity"],
        )
        layers_executed.append("hash_check")
        layer1_time = (time.time() - layer1_start) * 1000

        # ========== Layer 2: Metadata Analysis ==========
        layer2_start = time.time()
        metadata_data = self.metadata_service.analyze(image_bytes, filename)

        metadata_result = MetadataResult(
            has_c2pa=metadata_data.get("has_c2pa", False),
            c2pa_info=metadata_data.get("c2pa_info"),
            exif_data=metadata_data.get("exif_data"),
            ai_tool_signatures=metadata_data.get("ai_tool_signatures", []),
            software_used=metadata_data.get("software_used"),
            creation_date=metadata_data.get("creation_date"),
            exif_authenticity_score=metadata_data.get("exif_authenticity_score", 0.0),
            exif_inconsistencies=metadata_data.get("exif_inconsistencies", []),
        )
        layers_executed.append("metadata_analysis")
        layer2_time = (time.time() - layer2_start) * 1000

        # ========== Layer 3: AI Detection ==========
        detection_result = None
        layer3_time = 0

        layer3_start = time.time()
        detection_data = await self.detection_service.detect(image_bytes)

        if "error" not in detection_data:
            detection_result = DetectionResult(
                model_name=detection_data["model_name"],
                is_ai_generated=detection_data["is_ai_generated"],
                confidence=detection_data["confidence"],
                raw_scores=detection_data.get("raw_scores"),
            )
        layers_executed.append("ai_detection")
        layer3_time = (time.time() - layer3_start) * 1000

        # ========== Comprehensive Verdict ==========
        verdict, confidence, reasoning = self._compute_verdict(
            hash_result=hash_result,
            metadata_result=metadata_result,
            detection_result=detection_result,
        )

        total_time = (time.time() - start_time) * 1000

        # Create Result
        result = AnalysisResult(
            id=analysis_id,
            filename=filename,
            analyzed_at=datetime.utcnow(),
            hash_result=hash_result,
            metadata_result=metadata_result,
            detection_result=detection_result,
            final_verdict=verdict,
            confidence_score=confidence,
            reasoning=reasoning,
            total_execution_time_ms=round(total_time, 2),
            layers_executed=layers_executed,
        )
        return result

    def _compute_verdict(
        self,
        hash_result: HashResult,
        metadata_result: MetadataResult,
        detection_result: Optional[DetectionResult],
    ) -> tuple[VerdictType, float, str]:
        """
        Calculate Comprehensive Verdict

        Weight-based Verdict:
        - Hash: 30% (DinoV2 Similarity, Progressive Calculation)
        - Metadata: 40% (EXIF Authenticity + C2PA/Signature)
        - AI Detection: 30% (HuggingFace Model)

        Hash Progressive Calculation:
        - Above 85%: AI Score (Proportional to strength)
        - 70-85%: Uncertain Range (Score distributed)
        - Below 70%: Real Score
        """
        scores = {"ai": 0.0, "real": 0.0}
        reasons = []

        # 1. Hash-based Verdict (DinoV2 Vector Similarity) - Progressive Score
        similarity = hash_result.similarity

        if similarity >= 0.85:
            # Above 85%: Definite AI Image (Above threshold)
            ai_score = self.HASH_WEIGHT * min((similarity - 0.85) / 0.15 + 0.5, 1.0)
            scores["ai"] += ai_score
            reasons.append(
                f"⚠️ {'Matched' if hash_result.is_ai else 'High similarity'} with AI Image DB "
                f"(Similarity: {similarity:.1%})"
            )
        elif similarity >= 0.70:
            # 70-85%: Uncertain Range (Similar but low confidence)
            # Distribute score based on similarity
            uncertainty = (0.85 - similarity) / 0.15
            ai_portion = self.HASH_WEIGHT * 0.5 * (1 - uncertainty)
            real_portion = self.HASH_WEIGHT * 0.5 * uncertainty
            scores["ai"] += ai_portion
            scores["real"] += real_portion
            reasons.append(
                f"⚠️ Medium similarity with AI Image DB "
                f"(Similarity: {similarity:.1%}, Uncertain)"
            )
        else:
            # Below 70%: Likely Real Image
            real_score = self.HASH_WEIGHT * 0.5
            scores["real"] += real_score
            reasons.append(
                f"✓ Low similarity with AI Image DB (Max Similarity: {similarity:.1%})"
            )

        # 2. Metadata-based Verdict
        # 2-1. AI Tool Signature (Strong AI Evidence)
        if metadata_result.ai_tool_signatures:
            tools = ", ".join(metadata_result.ai_tool_signatures)
            scores["ai"] += self.METADATA_WEIGHT * 0.4
            reasons.append(f"🔍 AI Tool Signature Found: {tools}")

        # 2-2. C2PA Analysis
        if metadata_result.has_c2pa:
            c2pa_info = metadata_result.c2pa_info or {}
            if c2pa_info.get("ai_related_assertions"):
                scores["ai"] += self.METADATA_WEIGHT * 0.2
                reasons.append("🤖 AI generation info included in C2PA")
            else:
                # If C2PA exists but no AI info, likely real image
                scores["real"] += self.METADATA_WEIGHT * 0.15
                reasons.append("📜 C2PA Content Credentials present (No AI info)")

        # 2-3. Use EXIF Authenticity Score (Core Feature)
        exif_score = metadata_result.exif_authenticity_score

        if exif_score >= 0.7:
            # High EXIF Authenticity = Real Camera
            scores["real"] += self.METADATA_WEIGHT * 0.35 * exif_score
            reasons.append(
                f"📷 High EXIF Authenticity (Score: {exif_score:.2f}) - Likely Real Camera"
            )
        elif exif_score >= 0.3:
            # Medium Level
            scores["real"] += self.METADATA_WEIGHT * 0.15 * exif_score
            reasons.append(f"📷 EXIF Data Present (Authenticity: {exif_score:.2f})")
        else:
            # Low EXIF Authenticity = Suspected AI
            scores["ai"] += self.METADATA_WEIGHT * 0.25
            reasons.append(
                f"⚠️ Low EXIF Authenticity (Score: {exif_score:.2f}) - AI Generation Suspected"
            )

        # 2-4. EXIF Abnormal Pattern Detection
        if metadata_result.exif_inconsistencies:
            inconsistency_weight = min(
                len(metadata_result.exif_inconsistencies) * 0.05, 0.15
            )
            scores["ai"] += self.METADATA_WEIGHT * inconsistency_weight
            inconsistency_msgs = {
                "editing_software_without_camera": "Editing SW only",
                "perfect_square_ai_resolution": "AI Resolution",
                "unrealistic_aperture": "Unrealistic Settings",
                "missing_datetime_original": "Missing Original Time",
            }
            detected = [
                inconsistency_msgs.get(inc, inc)
                for inc in metadata_result.exif_inconsistencies
            ]
            reasons.append(f"⚠️ EXIF Abnormal Pattern: {', '.join(detected)}")

        # 3. AI Detection-based Verdict
        if detection_result:
            if detection_result.is_ai_generated:
                scores["ai"] += self.AI_DETECTION_WEIGHT * detection_result.confidence
                reasons.append(
                    f"🤖 AI Detection Model Verdict: AI Generated "
                    f"(Confidence: {detection_result.confidence:.1%})"
                )
            else:
                scores["real"] += self.AI_DETECTION_WEIGHT * detection_result.confidence
                reasons.append(
                    f"✅ AI Detection Model Verdict: Likely Real "
                    f"(Confidence: {detection_result.confidence:.1%})"
                )
        else:
            reasons.append("⏭️ AI Detection Skipped")

        # Final Verdict
        total_score = scores["ai"] + scores["real"]
        if total_score == 0:
            verdict = VerdictType.UNCERTAIN
            confidence = 0.5
        else:
            ai_ratio = scores["ai"] / total_score if total_score > 0 else 0.5

            if ai_ratio >= self.CONFIDENCE_THRESHOLD:
                verdict = VerdictType.AI_GENERATED
                confidence = ai_ratio
            elif ai_ratio <= (1 - self.CONFIDENCE_THRESHOLD):
                verdict = VerdictType.LIKELY_REAL
                confidence = 1 - ai_ratio
            else:
                verdict = VerdictType.UNCERTAIN
                confidence = 0.5 + abs(ai_ratio - 0.5)

        reasoning = " | ".join(reasons)

        return verdict, round(confidence, 4), reasoning
