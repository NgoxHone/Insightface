"""
Video Source Handler: Camera, RTSP, Webcam, or Video File

Provides unified interface for reading frames from various sources.
"""

import logging
from typing import Optional, Tuple
import cv2
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)


class VideoSource:
    """
    Unified video source handler

    Supports:
    - Webcam (device index 0, 1, ...)
    - RTSP streams (rtsp://...)
    - Video files (.mp4, .avi, .mov, etc.)
    - Image sequences (via pattern)
    """

    def __init__(
        self,
        source: str | int,
        width: int = None,
        height: int = None,
        fps: int = None,
        buffer_size: int = 1
    ):
        """
        Initialize video source

        Args:
            source: Source identifier
                - int: Webcam device index (0 for default)
                - str: RTSP URL or video file path
            width: Desired frame width (None = original)
            height: Desired frame height (None = original)
            fps: Desired FPS (for video files, None = original)
            buffer_size: OpenCV buffer size (for rtsp streams)
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.is_opened = False

        self._open()

    def _open(self):
        """Open video source"""
        try:
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.source}")

            # Set buffer size for RTSP streams
            if isinstance(self.source, str) and self.source.startswith('rtsp'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # Set resolution if specified
            if self.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Get actual properties
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.is_opened = True
            logger.info(
                f"Opened video source: {self.source} "
                f"({self.actual_width}x{self.actual_height} @ {self.actual_fps:.1f}fps)"
            )

        except Exception as e:
            logger.error(f"Failed to open video source {self.source}: {e}")
            self.is_opened = False
            raise

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read next frame

        Returns:
            (success, frame) tuple
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1

            # Resize if needed
            if self.width and self.height:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

        return ret, frame

    def get_properties(self) -> dict:
        """Get video source properties"""
        if not self.is_opened or self.cap is None:
            return {}

        return {
            'width': self.actual_width,
            'height': self.actual_height,
            'fps': self.actual_fps,
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fourcc': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            'mode': int(self.cap.get(cv2.CAP_PROP_MODE))
        }

    def is_ready(self) -> bool:
        """Check if source is ready for reading"""
        return self.is_opened and self.cap is not None

    def release(self):
        """Release video source"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logger.info(f"Released video source: {self.source}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()


def create_video_source(
    source: str | int = 0,
    resolution: str = "640x480",
    **kwargs
) -> VideoSource:
    """
    Factory function to create video source with common presets

    Args:
        source: Video source (0 for default webcam)
        resolution: "WxH" format (e.g., "640x480", "1280x720")
        **kwargs: Additional VideoSource args

    Returns:
        VideoSource instance
    """
    # Parse resolution
    if resolution:
        width, height = map(int, resolution.split('x'))
    else:
        width, height = 640, 480

    return VideoSource(
        source=source,
        width=width,
        height=height,
        **kwargs
    )
