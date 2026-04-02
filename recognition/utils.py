"""
Utility functions for face recognition
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def draw_face_annotations(
    image: np.ndarray,
    faces: List[Dict],
    draw_names: bool = True,
    draw_bbox: bool = True,
    draw_landmarks: bool = False,
    color_map: Optional[Dict] = None
) -> np.ndarray:
    """
    Draw bounding boxes, names, and landmarks on image

    Args:
        image: BGR image to annotate
        faces: List of face result dicts from recognize()
        draw_names: Draw person names
        draw_bbox: Draw bounding boxes
        draw_landmarks: Draw facial landmarks
        color_map: Dict mapping names to colors

    Returns:
        Annotated image
    """
    annotated = image.copy()

    for face in faces:
        bbox = face.get('bbox', [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Determine color based on match status
        name = face.get('name', 'unknown')
        matched = face.get('matched', False)
        confidence = face.get('confidence', 0.0)

        if color_map and name in color_map:
            color = color_map[name]
        else:
            if name == 'unknown':
                color = (0, 0, 255)  # Red for unknown
            elif matched and confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif matched:
                color = (0, 165, 255)  # Orange for medium confidence
            else:
                color = (0, 0, 255)  # Red

        # Draw bounding box
        if draw_bbox:
            thickness = 2 if matched else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw name and confidence
        if draw_names:
            label = f"{name}" if name == 'unknown' else f"{name}: {confidence:.2%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6 if len(label) < 20 else 0.5
            thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Background for text
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 4),
                (x1 + label_w, y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )

        # Draw landmarks
        if draw_landmarks and face.get('landmarks'):
            landmarks = np.array(face['landmarks'], dtype=int)
            for i, (lx, ly) in enumerate(landmarks):
                cv2.circle(annotated, (lx, ly), 2, (255, 0, 0), -1)

    return annotated


def crop_face(image: np.ndarray, bbox: List[int], margin: float = 0.2) -> np.ndarray:
    """
    Crop face from image with optional margin

    Args:
        image: Original image
        bbox: [x1, y1, x2, y2]
        margin: Additional margin around bbox (0-1)

    Returns:
        Cropped face image
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]

    # Add margin
    face_w = x2 - x1
    face_h = y2 - y1
    margin_x = int(face_w * margin)
    margin_y = int(face_h * margin)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    return image[y1:y2, x1:x2]


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    padding_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize image with padding to maintain aspect ratio

    Args:
        image: Input image
        target_size: (width, height)
        padding_color: Color for padding

    Returns:
        Resized image with padding
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute scale
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded canvas
    if len(image.shape) == 3:
        padded = np.full((target_h, target_w, 3), padding_color, dtype=resized.dtype)
    else:
        padded = np.full((target_h, target_w), padding_color, dtype=resized.dtype)

    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to enhance image contrast

    Args:
        image: BGR or grayscale image

    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

    return enhanced


def normalize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess image for ArcFace model

    Args:
        image: BGR face image
        target_size: Target resolution
        normalize: Normalize to [-1, 1]

    Returns:
        Preprocessed RGB image
    """
    # Resize
    if image.shape[:2] != target_size[::-1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize
    if normalize:
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5

    return image


def load_image_safe(path: str) -> Optional[np.ndarray]:
    """
    Load image from path with error handling

    Args:
        path: Image file path

    Returns:
        BGR image or None if failed
    """
    try:
        img = cv2.imread(path)
        if img is None or img.size == 0:
            logger.error(f"Failed to load image: {path}")
            return None
        return img
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None


def save_face_crop(
    face_image: np.ndarray,
    output_path: Path,
    quality: int = 95
) -> bool:
    """
    Save face crop to file

    Args:
        face_image: Face image (BGR)
        output_path: Output path
        quality: JPEG quality (1-100)

    Returns:
        True if successful
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(output_path),
            face_image,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save face crop: {e}")
        return False


def encode_image_to_bytes(
    image: np.ndarray,
    format: str = '.jpg',
    quality: int = 95
) -> Optional[bytes]:
    """
    Encode image to bytes for transmission

    Args:
        image: BGR image
        format: Image format (.jpg, .png)
        quality: Compression quality

    Returns:
        Encoded bytes or None
    """
    try:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality] if format == '.jpg' else []
        success, encoded = cv2.imencode(format, image, encode_param)
        if success:
            return encoded.tobytes()
        return None
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return None


def decode_image_from_bytes(data: bytes) -> Optional[np.ndarray]:
    """
    Decode image from bytes

    Args:
        data: Image bytes

    Returns:
        BGR image or None
    """
    try:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None
