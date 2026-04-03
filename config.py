"""
Configuration management for Face Recognition System
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Main configuration class"""

    # Project root
    PROJECT_ROOT: Path = Path(__file__).parent.resolve()

    # Paths
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    UPLOAD_DIR: Path = PROJECT_ROOT / "uploads"
    TRAINED_MODELS_DIR: Path = PROJECT_ROOT / "trained_models"
    STATIC_DIR: Path = PROJECT_ROOT / "static"

    # Database
    DATABASE_PATH: Path = DATA_DIR / "database.pkl"
    DATABASE_JSON_PATH: Path = DATA_DIR / "database.json"

    # Model settings
    MODEL_NAME: str = "buffalo_l"
    MODEL_PATH: Path = MODELS_DIR / MODEL_NAME
    PROVIDERS: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])

    # Detection
    DETECTION_THRESHOLD: float = 0.5
    MIN_FACE_SIZE: int = 20
    MAX_FACES_PER_IMAGE: int = 1000

    # Recognition
    RECOGNITION_THRESHOLD: float = 0.4  # cosine similarity threshold (0-1)
    MAX_EMBEDDINGS_PER_PERSON: int = 20
    USE_AVERAGE_EMBEDDING: bool = True  # Average multiple embeddings per person

    # Face alignment
    ENABLE_ALIGNMENT: bool = True
    ALIGNMENT_TARGET_SIZE: tuple = (112, 112)  # ArcFace standard

    # Quality checking
    ENABLE_QUALITY_CHECK: bool = True
    MIN_IMAGE_QUALITY_SCORE: float = 0.5  # 0-1 scale
    BLUR_THRESHOLD: float = 100.0  # Laplacian variance threshold
    MIN_FACE_PIXELS: int = 100 * 100  # Minimum face area
    MAX_YAW_ANGLE: float = 30.0  # Maximum head rotation
    MAX_PITCH_ANGLE: float = 25.0

    # Augmentation (for training)
    AUGMENTATION_CONFIG: dict = field(
        default_factory=lambda: {
            "flip_horizontal": True,
            "rotate_degrees": 10,
            "color_jitter": 0.2,
            "blur_prob": 0.1,
            "noise_prob": 0.1,
            "sharpness_range": (0.8, 1.2),
        }
    )

    # Training
    TRAIN_BATCH_SIZE: int = 32
    TRAIN_EPOCHS: int = 50
    TRAIN_LEARNING_RATE: float = 0.001
    TRAIN_VALIDATION_SPLIT: float = 0.2
    TRAIN_EARLY_STOPPING_PATIENCE: int = 10
    TRAIN_SAVE_CHECKPOINT_EVERY: int = 5

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5001
    API_DEBUG: bool = False
    API_THREADS: int = 4
    API_MAX_CONTENT_LENGTH: int = 50 * 1024 * 1024  # 50MB

    # Performance
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 300  # seconds
    BATCH_PROCESSING: bool = True
    MAX_BATCH_SIZE: int = 10

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = PROJECT_ROOT / "logs" / "face_recognition.log"

    # ========== Tracking Pipeline Settings ==========
    # YOLO Detection
    YOLO_MODEL_NAME: str = "yolov8n.pt"  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    YOLO_CONFIDENCE_THRESHOLD: float = 0.25  # Min confidence for detections
    YOLO_IOU_THRESHOLD: float = 0.45  # IOU threshold for NMS
    YOLO_DEVICE: str = "cpu"  # "cpu", "cuda", "cuda:0", etc.

    # Tracking (ByteTrack)
    TRACKING_ENABLED: bool = True
    TRACK_BUFFER: int = 30  # Frames to keep lost tracks (default 30 for 1 sec at 30fps)
    TRACK_CONFIDENCE_THRESHOLD: float = 0.1  # Lower threshold for tracking
    TRACK_IOU_THRESHOLD: float = 0.2  # IOU threshold for matching
    MATCH_THRESHOLD: float = 0.8  # Matching threshold for track association

    # Realtime Pipeline Performance
    FRAME_SKIP: int = 1  # Process every Nth frame for detection (1 = every frame)
    RECOGNITION_INTERVAL: float = 2.0  # Seconds between re-recognitions for same track
    MAX_TRACK_AGE: int = 120  # Maximum frames before track expires (~4 sec at 30fps)
    MIN_FACE_SIZE_FOR_RECOGNITION: int = 40  # Minimum face size (pixels) to run recognition

    # Face Extraction from Person BBox
    FACE_CROP_PADDING: float = 0.1  # Padding ratio around face region within person bbox
    FACE_SIZE_FOR_RECOGNITION: tuple = (112, 112)  # Size to resize face before recognition

    # Cache Settings
    TRACK_CACHE_TTL: int = 5  # Seconds to keep track→person mapping in cache
    MAX_CACHE_SIZE: int = 1000  # Maximum number of cached tracks

    def __post_init__(self):
        """Create necessary directories"""
        for dir_path in [
            self.MODELS_DIR,
            self.DATA_DIR,
            self.UPLOAD_DIR,
            self.TRAINED_MODELS_DIR,
            self.STATIC_DIR,
            self.DATA_DIR / "raw",
            self.DATA_DIR / "processed",
            self.DATA_DIR / "splits",
            self.UPLOAD_DIR / "temp",
            self.UPLOAD_DIR / "faces",
            self.MODELS_DIR / "checkpoints",
            self.MODELS_DIR / "finetuned",
            self.TRAINED_MODELS_DIR / "checkpoints",
            self.TRAINED_MODELS_DIR / "finetuned",
            self.TRAINED_MODELS_DIR / "logs",
            self.LOG_FILE.parent,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
