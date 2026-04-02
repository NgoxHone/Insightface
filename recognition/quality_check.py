"""
Face quality assessment module
Evaluates sharpness, brightness, pose, and other quality metrics
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

from config import config

logger = logging.getLogger(__name__)


def assess_face_quality(
    face_image: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    detailed: bool = False
) -> float:
    """
    Assess quality of a face image

    Args:
        face_image: Aligned or cropped face (BGR or grayscale)
        landmarks: 5-point facial landmarks if available
        detailed: Return detailed breakdown if True

    Returns:
        Quality score (0-1), or dict with breakdown if detailed=True
    """
    if face_image is None or face_image.size == 0:
        return 0.0

    scores = {}

    # 1. Sharpness / Blur detection
    sharpness = compute_sharpness(face_image)
    scores['sharpness'] = sharpness

    # 2. Size check
    size_score = compute_size_score(face_image)
    scores['size'] = size_score

    # 3. Brightness & Contrast
    brightness_score, contrast_score = compute_lighting_score(face_image)
    scores['brightness'] = brightness_score
    scores['contrast'] = contrast_score

    # 4. Pose estimation (if landmarks available)
    pose_score = 0.5
    if landmarks is not None and len(landmarks) >= 5:
        pose_score = compute_face_pose(landmarks, face_image.shape)
        scores['pose'] = pose_score

    # 5. Symmetry check
    symmetry_score = compute_symmetry(face_image)
    scores['symmetry'] = symmetry_score

    # Weighted average
    weights = {
        'sharpness': 0.30,
        'size': 0.15,
        'brightness': 0.15,
        'contrast': 0.10,
        'pose': 0.20,
        'symmetry': 0.10
    }

    total_weight = sum(weights.values())
    if detailed:
        weighted_sum = sum(scores[k] * weights[k] for k in scores if k in weights)
        overall = weighted_sum / total_weight
        return {
            'overall': float(np.clip(overall, 0, 1)),
            'breakdown': {k: float(v) for k, v in scores.items()}
        }

    weighted_sum = sum(scores[k] * weights[k] for k in scores if k in weights)
    overall = weighted_sum / total_weight

    return float(np.clip(overall, 0, 1))


def compute_sharpness(image: np.ndarray) -> float:
    """
    Compute sharpness using Laplacian variance
    Higher variance = sharper image

    Returns:
        0-1 score
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    # Normalize to 0-1 (empirical values)
    # variance < 50: very blurry
    # variance 100-500: acceptable
    # variance > 500: sharp
    score = min(variance / config.BLUR_THRESHOLD, 1.0)
    return max(0.0, score)


def compute_size_score(image: np.ndarray) -> float:
    """
    Score based on face image size (resolution)

    Args:
        image: Face image

    Returns:
        0-1 score
    """
    h, w = image.shape[:2]
    area = h * w

    if area >= 80 * 60:  # 4800 pixels
        return 1.0
    elif area >= 50 * 40:  # 2000 pixels
        return 0.7
    elif area >= 30 * 30:  # 900 pixels
        return 0.4
    else:
        return 0.1


def compute_lighting_score(image: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate brightness and contrast

    Returns:
        (brightness_score, contrast_score) both 0-1
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mean = np.mean(gray)
    std = np.std(gray)

    # Brightness: ideal around 100-180
    if 80 <= mean <= 180:
        brightness_score = 1.0
    elif 50 <= mean <= 220:
        brightness_score = 0.7
    else:
        brightness_score = 0.3

    # Contrast: higher std = better (up to a point)
    if std >= 40:
        contrast_score = 1.0
    elif std >= 25:
        contrast_score = 0.7
    elif std >= 15:
        contrast_score = 0.4
    else:
        contrast_score = 0.2

    return brightness_score, contrast_score


def compute_face_pose(landmarks: np.ndarray, image_shape: Tuple[int, ...]) -> float:
    """
    Estimate face pose from 5-point landmarks
    Returns score 0-1 (1 = perfectly frontal)

    Uses geometric heuristics for yaw and pitch estimation
    """
    try:
        h, w = image_shape[:2]

        # Landmarks order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]

        # 1. Yaw estimation: Compare eye x positions relative to face center
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_x = nose[0]
        yaw_deviation = abs(nose_x - eye_center_x) / w

        # If nose is centered between eyes, good profile
        yaw_score = max(0, 1.0 - yaw_deviation * 10)

        # 2. Pitch estimation: Check vertical alignment of eyes and mouth
        # In frontal face, eyes should be higher than nose, mouth lower
        left_eye_y = left_eye[1]
        right_eye_y = right_eye[1]
        nose_y = nose[1]
        left_mouth_y = left_mouth[1]
        right_mouth_y = right_mouth[1]

        eye_avg_y = (left_eye_y + right_eye_y) / 2
        mouth_avg_y = (left_mouth_y + right_mouth_y) / 2

        # Check if eyes are above nose and mouth below nose (within tolerance)
        eyes_above_nose = eye_avg_y < nose_y
        mouth_below_nose = mouth_avg_y > nose_y

        if eyes_above_nose and mouth_below_nose:
            pitch_score = 1.0
        else:
            pitch_score = 0.5

        # 3. Roll estimation: Eyes should be horizontal
        eye_y_diff = abs(left_eye_y - right_eye_y)
        eye_y_norm = eye_y_diff / h
        roll_score = max(0, 1.0 - eye_y_norm * 20)

        # Combine
        pose_score = (yaw_score * 0.5 + pitch_score * 0.3 + roll_score * 0.2)

        # Also consider symmetry
        left_mouth_x = left_mouth[0]
        right_mouth_x = right_mouth[0]
        mouth_center_x = (left_mouth_x + right_mouth_x) / 2
        nose_tip_x = nose[0]
        symmetry_deviation = abs(mouth_center_x - nose_tip_x) / w
        symmetry_score = max(0, 1.0 - symmetry_deviation * 20)

        final_score = 0.7 * pose_score + 0.3 * symmetry_score

        return float(np.clip(final_score, 0, 1))

    except Exception as e:
        logger.warning(f"Pose calculation failed: {e}")
        return 0.5  # Neutral score


def compute_symmetry(image: np.ndarray) -> float:
    """
    Check horizontal symmetry of face
    Frontal faces should be approximately symmetric

    Returns:
        0-1 score
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        # Flip horizontally
        flipped = cv2.flip(gray, 1)

        # Crop center to avoid hair/background differences
        margin_x = w // 10
        margin_y = h // 10

        left_half = gray[:, margin_x:-margin_x] if margin_x > 0 else gray
        right_half = flipped[:, margin_x:-margin_x] if margin_x > 0 else flipped

        # If sizes don't match due to odd dimensions, adjust
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]

        # Compute difference
        diff = cv2.absdiff(left_half, right_half)
        mean_diff = np.mean(diff)

        # Normalize: lower difference = higher symmetry
        # Typical difference values: 0-50 for symmetric faces
        score = max(0, 1.0 - mean_diff / 30.0)

        return float(score)

    except Exception as e:
        logger.warning(f"Symmetry calculation failed: {e}")
        return 0.5


def is_face_good_quality(
    face_image: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    threshold: float = None
) -> bool:
    """
    Quick check if face meets quality threshold

    Args:
        face_image: Face image
        landmarks: Facial landmarks
        threshold: Quality threshold (default from config)

    Returns:
        True if quality is acceptable
    """
    quality_threshold = threshold or config.MIN_IMAGE_QUALITY_SCORE
    quality = assess_face_quality(face_image, landmarks)
    return quality >= quality_threshold


def batch_assess_quality(
    face_images: List[np.ndarray],
    landmarks_list: Optional[List[np.ndarray]] = None
) -> List[float]:
    """
    Assess quality for multiple faces efficiently

    Args:
        face_images: List of face images
        landmarks_list: Corresponding landmarks (can be None)

    Returns:
        List of quality scores
    """
    scores = []
    for i, img in enumerate(face_images):
        lmk = landmarks_list[i] if landmarks_list and i < len(landmarks_list) else None
        score = assess_face_quality(img, lmk)
        scores.append(score)
    return scores
