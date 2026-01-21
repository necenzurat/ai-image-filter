"""
Layer 3: AI Detection Service
AI Generated Image Detection based on HuggingFace Models
"""

import io
import asyncio
from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import pipeline


class DetectionService:
    """AI Generated Image Detection Service"""

    # List of available models
    AVAILABLE_MODELS = {
        "Ateeqq/ai-vs-human-image-detector": {
            "description": "AI vs Real image classifier",
            "labels": {"artificial": "ai", "human": "real"},
        },
        "Organika/sdxl-detector": {
            "description": "SDXL generated image detector",
            "labels": {"artificial": "ai", "real": "real"},
        },
    }

    DEFAULT_MODEL = "Ateeqq/ai-vs-human-image-detector"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._classifier = None
        self._model_loaded = False

    @property
    def classifier(self):
        """Lazy loading of the classifier"""
        if self._classifier is None:
            self._load_model()
        return self._classifier

    def _load_model(self):
        """Load model (on first call)"""
        try:
            print(f"🔄 Loading model: {self.model_name}")

            # Check GPU availability
            device = 0 if torch.cuda.is_available() else -1

            self._classifier = pipeline(
                "image-classification", model=self.model_name, device=device
            )

            self._model_loaded = True
            print(f"✅ Model loaded successfully on {'GPU' if device == 0 else 'CPU'}")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    async def detect(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect if image is AI-generated

        Returns:
            - is_ai_generated: Boolean indicating AI generation
            - confidence: Confidence score (0.0 ~ 1.0)
            - raw_scores: Original scores
        """
        try:
            # Load image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Run inference (asynchronous)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, lambda: self.classifier(img))

            # Parse results
            return self._parse_results(results)

        except Exception as e:
            return {
                "model_name": self.model_name,
                "is_ai_generated": False,
                "confidence": 0.0,
                "error": str(e),
                "raw_scores": None,
            }

    def _parse_results(self, results: list) -> Dict[str, Any]:
        """Parse model results"""
        raw_scores = {r["label"]: r["score"] for r in results}

        # Sum scores for AI-related labels
        ai_score = 0.0
        real_score = 0.0

        for label, score in raw_scores.items():
            label_lower = label.lower()

            # AI related labels
            if any(
                ai_key in label_lower
                for ai_key in ["artificial", "ai", "fake", "generated", "synthetic"]
            ):
                ai_score += score
            # Real related labels
            elif any(
                real_key in label_lower
                for real_key in ["human", "real", "authentic", "natural", "hum"]
            ):
                real_score += score

        # Verdict
        is_ai = ai_score > real_score
        confidence = ai_score if is_ai else real_score

        return {
            "model_name": self.model_name,
            "is_ai_generated": is_ai,
            "confidence": round(confidence, 4),
            "raw_scores": raw_scores,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return current model information"""
        return {
            "model_name": self.model_name,
            "model_loaded": self._model_loaded,
            "available_models": list(self.AVAILABLE_MODELS.keys()),
            "device": "GPU" if torch.cuda.is_available() else "CPU",
        }
