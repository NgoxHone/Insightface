"""
Face Recognition Pipeline with YOLO Tracking

Real-time video processing with person tracking and face recognition.

Workflow:
1. Read frame from video source
2. YOLO tracking → person tracks with track IDs
3. InsightFace detection → faces with embeddings
4. Associate faces to tracks (by IOU overlap)
5. Track manager → map track ID to person identity (with caching)
6. Output results with tracking
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import cv2

from .detector import YOLODetector, get_detector
from .tracker import ByteTrackWrapper, get_tracker
from .track_manager import TrackManager, TrackRecord
from models.face_analyzer import face_analyzer
from recognition.database import FaceDatabase

from config import config

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    frame_skip: int = config.FRAME_SKIP
    recognition_interval: float = config.RECOGNITION_INTERVAL
    min_face_size: int = config.MIN_FACE_SIZE_FOR_RECOGNITION
    face_crop_padding: float = config.FACE_CROP_PADDING
    enable_tracking: bool = config.TRACKING_ENABLED
    max_track_age: int = config.MAX_TRACK_AGE
    iou_threshold_for_association: float = 0.5  # Min IOU to associate face with track


@dataclass
class TrackResult:
    """Result for a single track"""
    track_id: int
    bbox: List[int]
    person_id: str
    confidence: float
    face_bbox: Optional[List[int]] = None
    is_recognized: bool = False
    is_cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'person_id': self.person_id,
            'confidence': self.confidence,
            'face_bbox': self.face_bbox,
            'is_recognized': self.is_recognized,
            'is_cached': self.is_cached
        }


@dataclass
class PipelineOutput:
    """Pipeline output for a frame"""
    frame_id: int
    timestamp: float
    tracks: List[TrackResult]
    face_count: int
    track_count: int
    processing_time: float
    fps: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'tracks': [t.to_dict() for t in self.tracks],
            'face_count': self.face_count,
            'track_count': self.track_count,
            'processing_time': self.processing_time,
            'fps': self.fps
        }


class FaceTrackingPipeline:
    """
    Realtime face recognition pipeline with tracking

    Integrates YOLO detection/tracking with InsightFace recognition
    for efficient realtime video processing.
    """

    def __init__(
        self,
        database: FaceDatabase = None,
        config_override: PipelineConfig = None
    ):
        """
        Initialize pipeline

        Args:
            database: FaceDatabase instance (creates default if None)
            config_override: Override default configuration
        """
        # Initialize components
        self.detector = get_detector()  # YOLO detector
        self.tracker = get_tracker() if config.TRACKING_ENABLED else None
        self.track_manager = TrackManager()
        self.database = database or FaceDatabase()

        # Configuration
        self.cfg = config_override or PipelineConfig()

        # State
        self.frame_id = 0
        self.start_time = time.time()
        self.last_stats_log = 0

        logger.info("FaceTrackingPipeline initialized")
        logger.info(f"Config: frame_skip={self.cfg.frame_skip}, "
                   f"recognition_interval={self.cfg.recognition_interval}s")

    def process_frame(
        self,
        frame: np.ndarray,
        force_detect: bool = False
    ) -> PipelineOutput:
        """
        Process a single frame

        Args:
            frame: BGR numpy array
            force_detect: Force detection even if frame is skipped

        Returns:
            PipelineOutput with tracking results
        """
        start_time = time.time()
        self.frame_id += 1

        # Frame skipping for performance
        if not force_detect and self.frame_id % self.cfg.frame_skip != 0:
            # Return previous tracks without update (simple tracking)
            # In production, you might want to still run tracker on every frame
            # but skip detection. Here we skip entire processing.
            return self._get_empty_output()

        # Step 1: Person detection & tracking with YOLO
        tracks_raw = self._detect_and_track(frame)

        # Step 2: Face detection with InsightFace (on full frame)
        faces = self._detect_faces(frame)

        # Step 3: Associate faces with tracks
        track_face_map = self._associate_faces_to_tracks(tracks_raw, faces)

        # Step 4: Process each track (recognize or use cache)
        track_results = self._process_tracks(frame, tracks_raw, track_face_map, faces)

        # Step 5: Update track manager with new info
        for tr in track_results:
            track_record = self.track_manager.update_track(
                track_id=tr.track_id,
                bbox=tr.bbox,
                person_id=tr.person_id,
                embedding=self._get_face_embedding_for_track(tr, faces),
                confidence=tr.confidence
            )
            tr.is_cached = not track_record.needs_recognition(self.cfg.recognition_interval)

        # Step 6: Cleanup old tracks periodically
        if self.frame_id % 30 == 0:  # Every 30 frames
            self.track_manager.cleanup(max_age=self.cfg.max_track_age)

        # Calculate metrics
        processing_time = time.time() - start_time
        elapsed = time.time() - self.start_time
        fps = self.frame_id / elapsed if elapsed > 0 else 0

        # Log stats periodically
        if self.frame_id % 100 == 0:
            self._log_stats()

        output = PipelineOutput(
            frame_id=self.frame_id,
            timestamp=time.time(),
            tracks=track_results,
            face_count=len(faces),
            track_count=len(tracks_raw),
            processing_time=processing_time,
            fps=fps
        )

        return output

    def _detect_and_track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO tracking on frame"""
        if not self.cfg.enable_tracking or self.tracker is None:
            # Without tracking, just detect
            detections = self.detector.detect_persons(frame)
            # Convert detections to track-like format (temp IDs)
            tracks = []
            for i, det in enumerate(detections):
                tracks.append({
                    'track_id': -i - 1,  # Temporary negative IDs
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class_id': det['class_id'],
                    'class_name': det['class_name']
                })
            return tracks

        return self.tracker.update(frame)

    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces with InsightFace"""
        faces = face_analyzer.detect_and_align(frame)

        # Filter by minimum face size
        if self.cfg.min_face_size > 0:
            filtered = []
            for face in faces:
                bbox = face['bbox']
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w >= self.cfg.min_face_size and h >= self.cfg.min_face_size:
                    filtered.append(face)
            faces = filtered

        return faces

    def _associate_faces_to_tracks(
        self,
        tracks: List[Dict[str, Any]],
        faces: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Associate detected faces to person tracks using IOU

        Returns:
            Dict mapping track_id -> list of associated faces
        """
        track_face_map = {track['track_id']: [] for track in tracks}

        for face in faces:
            face_bbox = face['bbox']

            # Find track with highest IOU overlap
            best_track_id = None
            best_iou = 0

            for track in tracks:
                track_bbox = track['bbox']
                iou = self._calculate_iou(face_bbox, track_bbox)

                if iou > self.cfg.iou_threshold_for_association and iou > best_iou:
                    best_iou = iou
                    best_track_id = track['track_id']

            if best_track_id is not None:
                track_face_map[best_track_id].append(face)

        # Remove empty associations
        track_face_map = {tid: faces for tid, faces in track_face_map.items() if faces}

        return track_face_map

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y2_2)
        area_u = area_1 + area_2 - area_i

        return area_i / area_u if area_u > 0 else 0.0

    def _process_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        track_face_map: Dict[int, List[Dict[str, Any]]],
        all_faces: List[Dict[str, Any]]
    ) -> List[TrackResult]:
        """
        Process tracks: recognize or use cache

        Args:
            frame: Current frame (for debugging/visualization)
            tracks: List of tracks from tracker
            track_face_map: Mapping track_id -> associated faces
            all_faces: All detected faces

        Returns:
            List of TrackResult
        """
        results = []

        for track in tracks:
            track_id = track['track_id']
            track_bbox = track['bbox']

            # Check if this track has an associated face
            associated_faces = track_face_map.get(track_id, [])

            if not associated_faces:
                # No face visible for this track
                # Use cached person_id if available
                cached_record = self.track_manager.get_track_record(track_id)
                if cached_record:
                    person_id = cached_record.person_id
                    confidence = cached_record.confidence
                    is_recognized = False
                else:
                    person_id = "unknown"
                    confidence = 0.0
                    is_recognized = False
            else:
                # Use largest face for recognition
                face = max(
                    associated_faces,
                    key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1])
                )
                embedding = face.get('embedding')

                if embedding is None:
                    # No embedding, use cache
                    cached_record = self.track_manager.get_track_record(track_id)
                    if cached_record:
                        person_id = cached_record.person_id
                        confidence = cached_record.confidence
                        is_recognized = False
                    else:
                        person_id = "unknown"
                        confidence = 0.0
                        is_recognized = False
                else:
                    # Check if we need to recognize
                    needs_recog = self.track_manager.needs_recognition(track_id)

                    if needs_recog:
                        # Perform recognition
                        person_id, confidence = self.track_manager.match_by_embedding(
                            embedding, self.database
                        )
                        self.track_manager.update_track(
                            track_id=track_id,
                            bbox=track_bbox,
                            person_id=person_id,
                            embedding=embedding,
                            confidence=confidence
                        )
                        is_recognized = True
                        self.track_manager.stats['recognitions_performed'] += 1
                    else:
                        # Use cached person_id
                        cached_record = self.track_manager.get_track_record(track_id)
                        person_id = cached_record.person_id if cached_record else "unknown"
                        confidence = cached_record.confidence if cached_record else 0.0
                        is_recognized = False

                    # Set face_bbox for visualization
                    face_bbox = face['bbox']

            result = TrackResult(
                track_id=track_id,
                bbox=track_bbox,
                person_id=person_id,
                confidence=confidence,
                face_bbox=face_bbox if 'face_bbox' in locals() else None,
                is_recognized=is_recognized
            )
            results.append(result)

        return results

    def _get_face_embedding_for_track(
        self,
        track_result: TrackResult,
        all_faces: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Get face embedding for a track (for caching)"""
        if track_result.face_bbox:
            # Find the face with matching bbox
            for face in all_faces:
                if face['bbox'] == track_result.face_bbox:
                    return face.get('embedding')
        return None

    def _get_empty_output(self) -> PipelineOutput:
        """Return empty output for skipped frames"""
        return PipelineOutput(
            frame_id=self.frame_id,
            timestamp=time.time(),
            tracks=[],
            face_count=0,
            track_count=0,
            processing_time=0.0,
            fps=0.0
        )

    def _log_stats(self):
        """Log periodic statistics"""
        stats = self.track_manager.get_stats()
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_id / elapsed if elapsed > 0 else 0

        logger.info(
            f"Pipeline stats: frame={self.frame_id}, "
            f"fps={avg_fps:.1f}, "
            f"active_tracks={stats['active_tracks']}, "
            f"cache_hits={stats['cache_hits']}, "
            f"recognitions={stats['recognitions_performed']}, "
            f"occlusion_recoveries={stats['occlusion_recoveries']}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = self.track_manager.get_stats()
        stats['frame_id'] = self.frame_id
        stats['elapsed_time'] = time.time() - self.start_time
        stats['fps'] = self.frame_id / stats['elapsed_time'] if stats['elapsed_time'] > 0 else 0
        return stats

    def reset(self):
        """Reset pipeline state"""
        self.frame_id = 0
        self.start_time = time.time()
        if self.tracker:
            self.tracker.reset()
        self.track_manager = TrackManager()
        logger.info("Pipeline reset")

    def annotate_frame(
        self,
        frame: np.ndarray,
        output: PipelineOutput
    ) -> np.ndarray:
        """
        Draw tracking results on frame

        Args:
            frame: Original frame
            output: PipelineOutput from process_frame

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Colors for different persons
        colors = {}
        color_list = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
            (255, 0, 255), (255, 255, 0), (128, 128, 0), (0, 128, 128)
        ]

        for track in output.tracks:
            x1, y1, x2, y2 = track.bbox
            person_id = track.person_id

            # Get color for this person
            if person_id not in colors:
                colors[person_id] = color_list[len(colors) % len(color_list)]
            color = colors[person_id]

            # Draw track bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{person_id}:{track.track_id}"
            if track.is_cached:
                label += " [cached]"
            if track.is_recognized:
                label += " [new]"

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw face bbox if available
            if track.face_bbox:
                fx1, fy1, fx2, fy2 = track.face_bbox
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)

        # Draw stats
        stats_text = f"Frame: {output.frame_id} | FPS: {output.fps:.1f} | Tracks: {output.track_count}"
        cv2.putText(annotated, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return annotated
