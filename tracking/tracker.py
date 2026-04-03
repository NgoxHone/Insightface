"""
ByteTrack Tracker Wrapper for Object Tracking

Wraps ultralytics YOLO built-in tracking (which uses ByteTrack algorithm).
Provides simple interface for tracking objects across frames.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ultralytics import YOLO

from config import config

logger =logging.getLogger(__name__)


class ByteTrackWrapper:
    """
    ByteTrack tracker wrapper using ultralytics YOLO tracking

    Features:
    - High-performance tracking with re-identification
    - Handles occlusion and track recovery
    - Track state management (active, lost, removed)
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        track_buffer: int = None,
        match_threshold: float = None
    ):
        """
        Initialize ByteTrack tracker

        Args:
            model_name: YOLO model name/path
            device: Inference device
            confidence_threshold: Detection confidence threshold
            iou_threshold: IOU threshold for tracking
            track_buffer: Frames to keep lost tracks (default 30)
            match_threshold: Match threshold for track association (default 0.8)
        """
        self.model_name = model_name or config.YOLO_MODEL_NAME
        self.device = device or config.YOLO_DEVICE
        self.conf_threshold = confidence_threshold or config.TRACK_CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or config.TRACK_IOU_THRESHOLD
        self.track_buffer = track_buffer or config.TRACK_BUFFER
        self.match_threshold = match_threshold or config.MATCH_THRESHOLD

        self.model: Optional[YOLO] = None
        self._load_model()

        # Track history for debugging/analysis (optional)
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}

    def _load_model(self):
        """Load YOLO model with tracking capability"""
        try:
            logger.info(f"Loading YOLO model for tracking: {self.model_name}")
            self.model = YOLO(self.model_name)
            self.model.fuse()
            self.model.to(self.device)
            logger.info("YOLO tracker model loaded")
        except Exception as e:
            logger.error(f"Failed to load tracker model: {e}")
            raise

    def update(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Update tracks with new frame

        Args:
            image: Current frame (BGR numpy array)
            detections: Optional pre-computed detections (if None, run detection)
            **kwargs: Additional tracking parameters

        Returns:
            List of track dicts:
                - 'track_id': int (unique per object across frames)
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float
                - 'class_id': int
                - 'class_name': str
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to tracker update")
            return []

        try:
            # Run YOLO tracking
            # If detections provided, we'd need to manually feed them to tracker
            # For simplicity, let YOLO handle both detection and tracking
            results = self.model.track(
                image,
                persist=True,  # Keep tracks between calls
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                tracker="bytetrack.yaml",  # Use ByteTrack algorithm
                **kwargs
            )

            tracks = []
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None and hasattr(result.boxes, 'id'):
                    boxes = result.boxes.cpu().numpy()

                    for box in boxes:
                        # Get track ID (persistent across frames)
                        track_id = int(box.id[0]) if box.id is not None else -1
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id] if result.names else f"class_{class_id}"

                        # Update track history (for analysis)
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        self.track_history[track_id].append(center)

                        # Keep only last N points
                        max_history = 30
                        if len(self.track_history[track_id]) > max_history:
                            self.track_history[track_id].pop(0)

                        tracks.append({
                            'track_id': track_id,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        })

            logger.debug(f"Tracking: {len(tracks)} active tracks")
            return tracks

        except Exception as e:
            logger.error(f"Error in tracker update: {e}", exc_info=True)
            return []

    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Get movement history (center points) for a track"""
        return self.track_history.get(track_id, [])

    def reset(self):
        """Reset all tracks (start fresh)"""
        self.track_history.clear()
        # Reinitialize model to clear internal state
        logger.info("Tracker reset: all tracks cleared")

    def cleanup_old_tracks(self, max_age: int = None):
        """
        Remove old tracks from history to free memory

        Args:
            max_age: Maximum history length (default: config.MAX_TRACK_AGE)
        """
        if max_age is None:
            max_age = config.MAX_TRACK_AGE

        to_remove = []
        for track_id, history in self.track_history.items():
            if len(history) > max_age:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.track_history[track_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tracks")


# Convenience instance
byte_tracker: Optional[ByteTrackWrapper] = None


def get_tracker() -> ByteTrackWrapper:
    """Get or create global tracker instance"""
    global byte_tracker
    if byte_tracker is None:
        byte_tracker = ByteTrackWrapper()
    return byte_tracker
