"""
Model utilities for loading, saving, and managing face recognition models
"""
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2

from config import config

logger = logging.getLogger(__name__)


def check_cuda_available() -> bool:
    """Check if CUDA is available for ONNX Runtime"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_cuda = 'CUDAExecutionProvider' in providers
        logger.info(f"Available ONNX Runtime providers: {providers}")
        return has_cuda
    except ImportError:
        logger.warning("ONNX Runtime not installed")
        return False


def get_optimal_providers() -> list:
    """
    Get optimal execution providers based on available hardware

    Returns:
        List of provider names in priority order
    """
    if check_cuda_available():
        logger.info("CUDA available, using GPU acceleration")
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        logger.info("Using CPU execution")
        return ['CPUExecutionProvider']


def load_model(model_path: Optional[Path] = None, name: Optional[str] = None):
    """
    Load InsightFace model from path or name

    Args:
        model_path: Path to model directory
        name: Model name from InsightFace model zoo

    Returns:
        FaceAnalysis instance
    """
    from insightface.app import FaceAnalysis

    model_name = name or config.MODEL_NAME
    root_path = str(model_path) if model_path else str(config.MODELS_DIR)

    logger.info(f"Loading model: {model_name} from {root_path}")

    model = FaceAnalysis(name=model_name, root=root_path)
    model.prepare(
        ctx_id=0,
        det_thresh=config.DETECTION_THRESHOLD,
        det_size=(640, 640)
    )

    return model


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess image for model input

    Args:
        image: BGR numpy array
        target_size: Target size (width, height)
        normalize: Whether to normalize to [-1, 1]

    Returns:
        Preprocessed image
    """
    # Resize
    if image.shape[:2] != target_size[::-1]:  # (h, w) vs (w, h)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if normalize:
        # Normalize to [-1, 1] (standard for ArcFace)
        image = (image.astype(np.float32) - 127.5) / 127.5

    return image


def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings

    Args:
        emb1: First embedding (normalized)
        emb2: Second embedding (normalized)

    Returns:
        Cosine similarity score (0-1)
    """
    # Ensure embeddings are normalized
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)

    similarity = np.dot(emb1_norm, emb2_norm)
    return float(similarity)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.3
) -> List[int]:
    """
    Non-maximum suppression

    Args:
        boxes: Nx4 array of [x1, y1, x2, y2]
        scores: N array of confidence scores
        threshold: IoU threshold

    Returns:
        List of kept indices
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


def resize_image(
    image: np.ndarray,
    max_size: int = 640,
    keep_aspect: bool = True
) -> np.ndarray:
    """
    Resize image to fit within max_size

    Args:
        image: Input image
        max_size: Maximum dimension
        keep_aspect: Preserve aspect ratio

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if not keep_aspect:
        return cv2.resize(image, (max_size, max_size))

    if max(h, w) <= max_size:
        return image

    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h))


def compute_face_quality(
    image: np.ndarray,
    landmarks: Optional[np.ndarray] = None
) -> float:
    """
    Compute quality score for a face image

    Args:
        image: Face image (aligned or not)
        landmarks: 5-point landmarks if available

    Returns:
        Quality score (0-1)
    """
    scores = []

    # 1. Sharpness (Laplacian variance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / config.BLUR_THRESHOLD, 1.0)
    scores.append(sharpness_score)

    # 2. Size
    h, w = image.shape[:2]
    size_score = 1.0 if h >= 80 and w >= 60 else 0.5
    scores.append(size_score)

    # 3. Brightness
    if len(image.shape) == 3:
        brightness = np.mean(image)
        brightness_score = 1.0 if 50 < brightness < 200 else 0.3
        scores.append(brightness_score)

    # 4. Pose (if landmarks available)
    if landmarks is not None and len(landmarks) == 5:
        pose_score = compute_pose_score(landmarks, image.shape)
        scores.append(pose_score)

    # Weighted average
    quality = np.mean(scores)
    return float(np.clip(quality, 0, 1))


def compute_pose_score(
    landmarks: np.ndarray,
    image_shape: Tuple[int, int, ...]
) -> float:
    """
    Estimate head pose from 5-point landmarks

    Returns:
        Pose score (0-1), higher = more frontal
    """
    try:
        # Simple heuristic based on landmark positions
        # Nose tip, eye corners, mouth corners
        h, w = image_shape[:2]

        # Center of image
        center_x, center_y = w / 2, h / 2

        # Nose tip (landmark 2)
        nose = landmarks[2]
        nose_offset_x = abs(nose[0] - center_x) / (w / 2)
        nose_offset_y = abs(nose[1] - center_y) / (h / 2)

        # Eye corners (landmarks 0,1 and 3,4)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]

        # Eye line angle (horizontal if y's are similar)
        eye_angle = abs(left_eye[1] - right_eye[1]) / h

        # Combine heuristics
        pose_score = (1.0 - min(nose_offset_x, 1.0)) * 0.5 + (1.0 - min(eye_angle * 10, 1.0)) * 0.5

        return max(0, pose_score)
    except Exception:
        return 0.5  # Default if calculation fails


def ensure_model_downloaded(model_name: str = None) -> Path:
    """
    Ensure model is downloaded and return path

    Args:
        model_name: Name of model in InsightFace zoo

    Returns:
        Path to model directory
    """
    from insightface.model_zoo import get_model

    name = model_name or config.MODEL_NAME
    model_dir = config.MODELS_DIR

    try:
        # This will download if not exists
        model_path = get_model(name=name, root=str(model_dir))
        logger.info(f"Model available at: {model_path}")
        return Path(model_path)
    except Exception as e:
        logger.error(f"Failed to download model {name}: {e}")
        raise
