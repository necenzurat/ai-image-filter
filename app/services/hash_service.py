"""
Layer 1: Hash Service
Image Hash Calculation and Duplication Check
"""

import io
from typing import Dict, Optional, Tuple
from PIL import Image
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel


class HashService:
    """Image Hash Calculation and Duplication Check Service"""

    def __init__(
        self,
        db_vectors_path: str = "./data/ai_dinohashes.npy",
        metadata_path: str = "./data/ai_metadata.csv",
        threshold: float = 0.85,
    ):
        """
        HashService

        Args:
            db_vectors_path: Path to AI image vector file
            metadata_path: Path to AI image metadata file
            threshold: Similarity threshold (0~1)
        """
        # Load DinoV2 Model
        self.model_name = "facebook/dinov2-small"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load DB vectors and metadata
        self.db_vectors = np.load(db_vectors_path)
        self.metadata = pd.read_csv(metadata_path)
        self.threshold = threshold

    def _extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract feature vectors from image using DinoV2 model

        Args:
            image: PIL Image object

        Returns:
            Feature vector (numpy array)
        """
        # Image Preprocessing
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Feature Extraction (no gradient calculation)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use output of CLS token
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return features.flatten()

    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: Vector 1
            vec2: Vector 2

        Returns:
            Cosine similarity (0~1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def find_similar_image(
        self, image_vector: np.ndarray
    ) -> Tuple[Optional[int], float]:
        """
        Find the most similar image in the DB

        Args:
            image_vector: Vector of the image to search

        Returns:
            (Index, Similarity) tuple. (None, 0.0) if no similar image found.
        """
        max_similarity = 0.0
        max_idx = None

        for idx, db_vector in enumerate(self.db_vectors):
            similarity = self._compute_cosine_similarity(image_vector, db_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                max_idx = idx

        if max_similarity >= self.threshold:
            return max_idx, max_similarity
        return None, max_similarity

    def compute_hash(self, image_bytes: bytes) -> Dict[str, any]:
        """
        Calculate dinohash of image and determine if it is an AI image

        Args:
            image_bytes: Image byte data

        Returns:
            - is_ai: Boolean indicating AI image
            - similarity: Maximum similarity score
        """
        # Load Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Extract feature vectors with DinoV2
        image_vector = self._extract_features(image)

        # Find similar image in DB
        matched_idx, similarity = self.find_similar_image(image_vector)

        result = {
            "is_ai": matched_idx is not None,
            "similarity": float(similarity),
        }

        return result
