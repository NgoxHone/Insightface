"""
Face Analysis Wrapper using InsightFace
Provides face detection, alignment, and embedding extraction
"""

import threading
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import cv2
import insightface
from insightface.model_zoo import get_model
from insightface.utils import face_align

from config import config

logger = logging.getLogger(__name__)


class FaceAnalysisSingleton:
    """Singleton wrapper for InsightFace FaceAnalysis to avoid multiple loads"""

    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load the InsightFace model"""
        try:
            logger.info(f"Loading InsightFace model: {config.MODEL_NAME}")
            self._model = insightface.app.FaceAnalysis(
                name=config.MODEL_NAME,
                providers=config.PROVIDERS,
                root=str(config.MODELS_DIR),
            )
            self._model.prepare(
                ctx_id=0,
                det_thresh=config.DETECTION_THRESHOLD,
                det_size=(640, 640),  # Can be tuned
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @property
    def model(self):
        """Get the loaded model"""
        if self._model is None:
            self._load_model()
        return self._model

    def reload(self):
        """Reload the model (useful after fine-tuning)"""
        self._model = None
        self._load_model()

    def detect_faces(self, image: np.ndarray) -> List[Any]:
        """
        Detect faces in image

        Args:
            image: BGR numpy array (OpenCV format)

        Returns:
            List of Face objects with bbox, landmarks, etc.
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect_faces")
            return []

        try:
            faces = self.model.get(image)
            # Filter by minimum face size
            filtered_faces = []
            for face in faces:
                bbox = face.bbox.astype(int)
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                if (
                    face_width >= config.MIN_FACE_SIZE
                    and face_height >= config.MIN_FACE_SIZE
                ):
                    filtered_faces.append(face)
                else:
                    logger.debug(f"Filtered out small face: {face_width}x{face_height}")

            logger.debug(f"Detected {len(filtered_faces)} faces")
            return filtered_faces[: config.MAX_FACES_PER_IMAGE]
        except Exception as e:
            logger.error(f"Error in detect_faces: {e}")
            return []

    def get_embedding(self, face: Any, aligned: bool = False) -> Optional[np.ndarray]:
        """
        Get embedding vector for a face

        Args:
            face: Face object from detect_faces
            aligned: Whether face is already aligned

        Returns:
            512-D normalized embedding vector or None
        """
        try:
            if aligned:
                # Use normed_embedding if face is already aligned
                if hasattr(face, "normed_embedding"):
                    return face.normed_embedding
            else:
                # Get embedding from face object
                if hasattr(face, "embedding"):
                    # Normalize embedding to unit length
                    embedding = face.embedding
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        return embedding / norm
            return None
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align face using 5-point landmarks

        Args:
            image: Original BGR image
            landmarks: 5x2 array of (x, y) coordinates

        Returns:
            Aligned face image (112x112 by default)
        """
        try:
            aligned = face_align.norm_crop(
                image, landmark=landmarks, image_size=config.ALIGNMENT_TARGET_SIZE[0]
            )
            return aligned
        except Exception as e:
            logger.error(f"Error aligning face: {e}")
            # Return resized original if alignment fails
            return cv2.resize(image, config.ALIGNMENT_TARGET_SIZE)

    def detect_and_align(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces and return aligned crops with metadata

        Args:
            image: BGR numpy array

        Returns:
            List of dicts with keys:
                - 'face_img': aligned face image
                - 'bbox': [x1, y1, x2, y2]
                - 'landmarks': 5x2 array
                - 'face': original face object
                - 'quality_score': float
        """
        faces = self.detect_faces(image)
        results = []

        for face in faces:
            try:
                bbox = face.bbox.astype(int)
                landmarks = face.kps  # 5 keypoints

                # Align face
                if config.ENABLE_ALIGNMENT and landmarks is not None:
                    aligned_face = self.align_face(image, landmarks)
                else:
                    # Crop without alignment
                    x1, y1, x2, y2 = bbox
                    aligned_face = cv2.resize(
                        image[y1:y2, x1:x2], config.ALIGNMENT_TARGET_SIZE
                    )

                # Get embedding
                embedding = self.get_embedding(face, aligned=True)

                result = {
                    "face_img": aligned_face,
                    "bbox": bbox.tolist(),
                    "landmarks": landmarks.tolist() if landmarks is not None else None,
                    "embedding": embedding,
                    "face_obj": face,
                }
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing a detected face: {e}")
                continue

        return results

    def get_face_embedding_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect largest face and return its embedding
        Convenience method for single-face registration

        Args:
            image: BGR numpy array

        Returns:
            Normalized 512-D embedding or None if no face found
        """
        faces = self.detect_faces(image)
        if not faces:
            return None

        # Select largest face
        largest_face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        return self.get_embedding(largest_face, aligned=True)

    def warmup(self):
        """Warm up the model with a dummy inference"""
        logger.info("Warming up model...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect_faces(dummy_img)
        logger.info("Model warmed up")


# Convenience singleton instance
face_analyzer = FaceAnalysisSingleton()
