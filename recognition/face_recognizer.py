"""
Face Recognizer: Main recognition pipeline (detect + align + recognize)
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from models.face_analyzer import face_analyzer
from .database import FaceDatabase
from .quality_check import assess_face_quality
from .utils import draw_face_annotations

from config import config

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Main face recognition pipeline

    Workflow:
        1. Detect faces in image
        2. (Optional) Check quality
        3. Align faces
        4. Extract embeddings
        5. Search database for matches
    """

    def __init__(
        self,
        database_path: Optional[Path] = None,
        threshold: float = None,
        enable_quality_check: bool = None
    ):
        """
        Initialize face recognizer

        Args:
            database_path: Path to face database pickle file
            threshold: Recognition threshold (overrides config)
            enable_quality_check: Override config setting
        """
        self.database = FaceDatabase(database_path or config.DATABASE_PATH)
        self.threshold = threshold or config.RECOGNITION_THRESHOLD
        self.enable_quality_check = enable_quality_check if enable_quality_check is not None else config.ENABLE_QUALITY_CHECK

        # Warmup model
        face_analyzer.warmup()

        logger.info(f"FaceRecognizer initialized: threshold={self.threshold}, quality_check={self.enable_quality_check}")

    def set_threshold(self, threshold: float):
        """Update recognition threshold"""
        self.threshold = threshold
        logger.info(f"Threshold updated to {threshold}")

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image (without recognition)

        Args:
            image: BGR numpy array

        Returns:
            List of face info dicts with bbox, landmarks, quality_score
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect")
            return []

        results = face_analyzer.detect_and_align(image)
        faces_info = []

        for result in results:
            face_img = result['face_img']
            landmarks = result.get('landmarks')

            # Quality assessment
            quality_score = 0.5
            if self.enable_quality_check:
                quality_score = assess_face_quality(face_img, landmarks)

            face_info = {
                'bbox': result['bbox'],
                'landmarks': result['landmarks'],
                'quality_score': float(quality_score),
                'face_size': face_img.shape[:2] if face_img is not None else None
            }
            faces_info.append(face_info)

        logger.debug(f"Detected {len(faces_info)} faces")
        return faces_info

    def recognize(
        self,
        image: np.ndarray,
        threshold: float = None,
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        Identify faces in image

        Args:
            image: BGR numpy array
            threshold: Override threshold for this call
            return_all: Return all candidates (not just top match)

        Returns:
            Dictionary with faces and recognition results
        """
        if threshold is None:
            threshold = self.threshold

        if image is None or image.size == 0:
            return {'success': False, 'error': 'Empty image', 'faces': []}

        try:
            # Detect and align faces
            face_results = face_analyzer.detect_and_align(image)

            if not face_results:
                return {'success': True, 'faces': [], 'message': 'No faces detected'}

            faces_output = []
            for idx, result in enumerate(face_results):
                embedding = result.get('embedding')
                bbox = result['bbox']
                landmarks = result['landmarks']

                if embedding is None:
                    logger.warning(f"Face {idx} has no embedding")
                    continue

                # Search database
                matches = self.database.search(embedding, threshold=threshold, top_k=5 if return_all else 1)

                # Quality check
                face_img = result['face_img']
                quality_score = assess_face_quality(face_img, landmarks) if self.enable_quality_check else 1.0

                face_output = {
                    'id': idx,
                    'bbox': bbox,
                    'landmarks': landmarks,
                    'quality_score': float(quality_score),
                    'embedding_extracted': True
                }

                if matches:
                    best_match, confidence = matches[0]
                    face_output.update({
                        'name': best_match,
                        'confidence': float(confidence),
                        'matched': True,
                        'candidates': [
                            {'name': name, 'confidence': float(conf)}
                            for name, conf in matches
                        ] if return_all else []
                    })
                else:
                    face_output.update({
                        'name': 'unknown',
                        'confidence': 0.0,
                        'matched': False,
                        'candidates': []
                    })

                faces_output.append(face_output)

            return {
                'success': True,
                'faces': faces_output,
                'total_faces': len(faces_output)
            }

        except Exception as e:
            logger.error(f"Error in recognize: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'faces': []}

    def identify_single_face(
        self,
        face_crop: np.ndarray,
        threshold: float = None
    ) -> Optional[Tuple[str, float]]:
        """
        Recognize a single face crop

        Args:
            face_crop: Already cropped face image (aligned or not)
            threshold: Recognition threshold

        Returns:
            (name, confidence) or None
        """
        if threshold is None:
            threshold = self.threshold

        try:
            embedding = face_analyzer.get_embedding_from_image(face_crop)
            if embedding is None:
                return None

            matches = self.database.search(embedding, threshold=threshold, top_k=1)
            if matches:
                return matches[0]
            return ('unknown', 0.0)
        except Exception as e:
            logger.error(f"Error in identify_single_face: {e}")
            return None

    def register_person(
        self,
        name: str,
        images: List[np.ndarray],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Register a new person with multiple face images

        Args:
            name: Person's name
            images: List of face images (BGR numpy arrays)
            metadata: Optional additional data

        Returns:
            Registration result dictionary
        """
        if not name or not name.strip():
            return {'success': False, 'error': 'Invalid name', 'embeddings_extracted': 0}

        if not images:
            return {'success': False, 'error': 'No images provided', 'embeddings_extracted': 0}

        embeddings = []
        quality_scores = []

        for idx, img in enumerate(images):
            if img is None or img.size == 0:
                logger.warning(f"Skipping invalid image at index {idx}")
                continue

            # Detect and align largest face
            face_results = face_analyzer.detect_and_align(img)
            if not face_results:
                logger.warning(f"No face detected in image {idx}")
                continue

            # Use largest face
            largest_face = max(
                face_results,
                key=lambda r: (r['bbox'][2] - r['bbox'][0]) * (r['bbox'][3] - r['bbox'][1])
            )

            embedding = largest_face.get('embedding')
            if embedding is None:
                logger.warning(f"No embedding for face in image {idx}")
                continue

            # Quality check
            quality_score = 1.0
            if self.enable_quality_check:
                quality_score = assess_face_quality(largest_face['face_img'], largest_face['landmarks'])
                if quality_score < config.MIN_IMAGE_QUALITY_SCORE:
                    logger.warning(f"Image {idx} has low quality ({quality_score:.3f}), skipping")
                    continue

            embeddings.append(embedding)
            quality_scores.append(quality_score)

        if not embeddings:
            return {
                'success': False,
                'error': 'No valid faces extracted from images',
                'embeddings_extracted': 0,
                'average_quality': 0.0
            }

        # Add to database
        success = self.database.add_person(name, embeddings, metadata)

        avg_quality = np.mean(quality_scores) if quality_scores else 0.0

        return {
            'success': success,
            'name': name,
            'embeddings_extracted': len(embeddings),
            'average_quality': float(avg_quality),
            'total_embeddings_for_person': len(self.database.get_person(name)['embeddings']) if success else 0
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        stats = self.database.get_stats()
        stats['threshold'] = self.threshold
        stats['quality_check_enabled'] = self.enable_quality_check
        return stats

    def draw_annotations(
        self,
        image: np.ndarray,
        faces_output: List[Dict],
        draw_names: bool = True,
        draw_bbox: bool = True,
        draw_landmarks: bool = False
    ) -> np.ndarray:
        """
        Draw recognition results on image

        Args:
            image: Original BGR image
            faces_output: Output from recognize()
            draw_names: Draw person names
            draw_bbox: Draw bounding boxes
            draw_landmarks: Draw facial landmarks

        Returns:
            Annotated image
        """
        return draw_face_annotations(
            image, faces_output,
            draw_names=draw_names,
            draw_bbox=draw_bbox,
            draw_landmarks=draw_landmarks
        )
