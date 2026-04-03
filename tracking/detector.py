"""
YOLOv8 Detector Wrapper for Person Detection

Provides person detection optimized for face recognition pipeline.
Can also detect faces if configured (using custom YOLO face model).
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from ultralytics import YOLO

from config import config

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLOv8 detector wrapper for person detection

    Supports:
    - Person detection (COCO class 0)
    - Optional face detection (if custom model provided)
    - Batch processing
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        classes: List[int] = None  # Filter by class IDs (0 = person)
    ):
        """
        Initialize YOLO detector

        Args:
            model_name: YOLO model name/path (e.g., 'yolov8n.pt', 'yolov8s.pt')
            device: Device for inference ('cpu', 'cuda', 'cuda:0')
            confidence_threshold: Min detection confidence
            iou_threshold: NMS IOU threshold
            classes: List of class IDs to detect (None = all classes)
        """
        self.model_name = model_name or config.YOLO_MODEL_NAME
        self.device = device or config.YOLO_DEVICE
        self.conf_threshold = confidence_threshold or config.YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or config.YOLO_IOU_THRESHOLD

        # Default to person detection only (COCO class 0)
        self.classes = classes if classes is not None else [0]

        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {self.model_name} on device: {self.device}")
            self.model = YOLO(self.model_name)
            self.model.fuse()
            self.model.to(self.device)
            logger.info(f"YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        classes: List[int] = None,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in image

        Args:
            image: BGR numpy array (OpenCV format)
            confidence_threshold: Override default confidence threshold
            iou_threshold: Override default IOU threshold
            classes: Override default class filter
            verbose: Enable verbose logging

        Returns:
            List of detection dicts with keys:
                - 'bbox': [x1, y1, x2, y2] (int)
                - 'confidence': float
                - 'class_id': int
                - 'class_name': str
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect")
            return []

        conf = confidence_threshold if confidence_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        cls = classes if classes is not None else self.classes

        try:
            # Run inference
            results = self.model(
                image,
                conf=conf,
                iou=iou,
                classes=cls,
                verbose=verbose,
                device=self.device
            )

            detections = []
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id] if result.names else f"class_{class_id}"

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        })

            logger.debug(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Error in YOLO detect: {e}", exc_info=True)
            return []

    def detect_persons(
        self,
        image: np.ndarray,
        min_size: Tuple[int, int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect only persons (convenience method)

        Args:
            image: BGR numpy array
            min_size: Minimum (width, height) for person bbox to be valid
            **kwargs: Additional args for detect()

        Returns:
            List of person detection dicts (class_id=0)
        """
        detections = self.detect(image, classes=[0], **kwargs)

        # Filter by minimum size if specified
        if min_size:
            min_w, min_h = min_size
            filtered = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                w, h = x2 - x1, y2 - y1
                if w >= min_w and h >= min_h:
                    filtered.append(det)
            detections = filtered

        return detections

    def warmup(self, input_size: Tuple[int, int] = (640, 640)):
        """Warm up model with dummy inference"""
        logger.info("Warming up YOLO model...")
        dummy = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        self.detect(dummy, verbose=False)
        logger.info("YOLO model warmed up")


# Convenience singleton instance
yolo_detector: Optional[YOLODetector] = None


def get_detector() -> YOLODetector:
    """Get or create global detector instance"""
    global yolo_detector
    if yolo_detector is None:
        yolo_detector = YOLODetector()
    return yolo_detector
