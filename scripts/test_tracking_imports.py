#!/usr/bin/env python3
"""
Quick test to verify all tracking module imports work correctly
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing tracking module imports...")

try:
    from tracking.detector import YOLODetector, get_detector
    print("✓ detector module")
except Exception as e:
    print(f"✗ detector module: {e}")

try:
    from tracking.tracker import ByteTrackWrapper, get_tracker
    print("✓ tracker module")
except Exception as e:
    print(f"✗ tracker module: {e}")

try:
    from tracking.track_manager import TrackManager, TrackRecord
    print("✓ track_manager module")
except Exception as e:
    print(f"✗ track_manager module: {e}")

try:
    from tracking.video_source import VideoSource, create_video_source
    print("✓ video_source module")
except Exception as e:
    print(f"✗ video_source module: {e}")

try:
    from tracking.pipeline import FaceTrackingPipeline, PipelineConfig
    print("✓ pipeline module")
except Exception as e:
    print(f"✗ pipeline module: {e}")

try:
    from recognition.database import FaceDatabase
    print("✓ FaceDatabase import")
except Exception as e:
    print(f"✗ FaceDatabase: {e}")

try:
    from models.face_analyzer import face_analyzer
    print("✓ face_analyzer import")
except Exception as e:
    print(f"✗ face_analyzer: {e}")

print("\nAll imports tested!")
