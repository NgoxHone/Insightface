"""
API schemas and validation
"""
from typing import Optional, List
from dataclasses import dataclass, asdict
from flask import Request
import werkzeug


@dataclass
class APIResponse:
    """Standard API response format"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'data': self.data or {},
            'error': self.error
        }


@dataclass
class RegisterRequest:
    """Schema for registration request"""
    name: str

    MIN_NAME_LENGTH = 1
    MAX_NAME_LENGTH = 100

    @classmethod
    def validate(cls, name: str, images: List) -> Optional[str]:
        """Validate registration request, return error message or None"""
        if not name or not name.strip():
            return "Name is required"
        if len(name) < cls.MIN_NAME_LENGTH:
            return f"Name must be at least {cls.MIN_NAME_LENGTH} character"
        if len(name) > cls.MAX_NAME_LENGTH:
            return f"Name must be at most {cls.MAX_NAME_LENGTH} characters"
        if not images or len(images) == 0:
            return "At least one image is required"
        return None


@dataclass
class RecognizeRequest:
    """Schema for recognition request"""
    threshold: float = 0.6
    MIN_THRESHOLD = 0.0
    MAX_THRESHOLD = 1.0

    @classmethod
    def validate(cls, threshold: Optional[float]) -> Optional[str]:
        if threshold is not None:
            if not (cls.MIN_THRESHOLD <= threshold <= cls.MAX_THRESHOLD):
                return f"Threshold must be between {cls.MIN_THRESHOLD} and {cls.MAX_THRESHOLD}"
        return None


def extract_files_from_request(request: Request, field_name: str = 'images') -> List:
    """
    Extract files from multipart/form-data request

    Args:
        request: Flask request object
        field_name: Name of the file field

    Returns:
        List of file objects
    """
    if field_name in request.files:
        files = request.files.getlist(field_name)
        return [f for f in files if f and f.filename and f.filename != '']
    return []


def extract_image_from_request(request: Request, field_name: str = 'image'):
    """Extract single image from request"""
    if field_name in request.files:
        file = request.files[field_name]
        if file and file.filename:
            return file
    return None


def parse_float(value, default=None):
    """Safely parse float from request"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
