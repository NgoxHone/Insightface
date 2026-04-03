"""
Track Manager: Maps track IDs to person identities with caching

Responsibilities:
- Track ID ↔ Person ID mapping
- Cache recognition results to avoid redundant computation
- Handle track reappearance after occlusion
- Clean up stale tracks
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np

from config import config

logger = logging.getLogger(__name__)


@dataclass
class TrackRecord:
    """Track record data"""
    track_id: int
    person_id: str = "unknown"
    last_seen: float = 0.0
    last_recognized: float = 0.0
    embedding: Optional[np.ndarray] = None
    confidence: float = 0.0
    bbox: List[int] = field(default_factory=list)

    def is_expired(self, ttl: float = None) -> bool:
        """Check if track record is expired"""
        if ttl is None:
            ttl = config.TRACK_CACHE_TTL
        return (time.time() - self.last_seen) > ttl

    def needs_recognition(self, recognition_interval: float = None) -> bool:
        """Check if track needs re-recognition (refresh)"""
        interval = recognition_interval or config.RECOGNITION_INTERVAL
        return (time.time() - self.last_recognized) > interval


class TrackManager:
    """
    Manages track-to-person mapping with caching

    Optimizations:
    - Cache recognition results per track
    - Refresh recognition periodically
    - Handle track ID changes (occlusion recovery)
    - LRU eviction for memory management
    """

    def __init__(
        self,
        cache_ttl: float = None,
        recognition_interval: float = None,
        max_cache_size: int = None
    ):
        """
        Initialize track manager

        Args:
            cache_ttl: Time-to-live for cached track records (seconds)
            recognition_interval: Min time between re-recognition for same track
            max_cache_size: Maximum number of tracks to keep in cache
        """
        self.cache_ttl = cache_ttl or config.TRACK_CACHE_TTL
        self.recognition_interval = recognition_interval or config.RECOGNITION_INTERVAL
        self.max_cache_size = max_cache_size or config.MAX_CACHE_SIZE

        # Track records: track_id -> TrackRecord
        self._tracks: Dict[int, TrackRecord] = {}

        # Person ID to track IDs mapping (for debugging/monitoring)
        self._person_tracks: Dict[str, List[int]] = {}

        # Statistics
        self.stats = {
            'total_tracks_seen': 0,
            'recognitions_performed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'occlusion_recoveries': 0
        }

        logger.info(
            f"TrackManager initialized: cache_ttl={self.cache_ttl}s, "
            f"recognition_interval={self.recognition_interval}s, "
            f"max_cache_size={self.max_cache_size}"
        )

    def update_track(
        self,
        track_id: int,
        bbox: List[int],
        person_id: str = None,
        embedding: np.ndarray = None,
        confidence: float = 0.0
    ) -> TrackRecord:
        """
        Update or create track record

        Args:
            track_id: Track ID from tracker
            bbox: Bounding box [x1, y1, x2, y2]
            person_id: Recognized person ID (None if not recognized yet)
            embedding: Face embedding (for re-identification)
            confidence: Recognition confidence

        Returns:
            Updated TrackRecord
        """
        current_time = time.time()

        if track_id in self._tracks:
            # Update existing track
            record = self._tracks[track_id]
            record.last_seen = current_time
            record.bbox = bbox

            # Update person_id if provided (new recognition)
            if person_id is not None:
                old_person_id = record.person_id
                record.person_id = person_id
                record.last_recognized = current_time
                record.confidence = confidence
                if embedding is not None:
                    record.embedding = embedding

                # Update person→tracks mapping if changed
                if old_person_id != person_id:
                    self._update_person_tracks_mapping(old_person_id, person_id, track_id)

            logger.debug(f"Updated track {track_id}: {record.person_id}")
        else:
            # Create new track record
            record = TrackRecord(
                track_id=track_id,
                person_id=person_id or "unknown",
                last_seen=current_time,
                last_recognized=current_time if person_id else 0.0,
                bbox=bbox,
                embedding=embedding,
                confidence=confidence
            )
            self._tracks[track_id] = record
            self.stats['total_tracks_seen'] += 1

            if person_id and person_id != "unknown":
                self._add_person_track_mapping(person_id, track_id)

            logger.debug(f"Created new track {track_id}: {person_id or 'unrecognized'}")

        # Enforce cache size limit (LRU eviction)
        if len(self._tracks) > self.max_cache_size:
            self._evict_oldest_tracks()

        return record

    def get_person_id(self, track_id: int) -> Optional[str]:
        """
        Get person ID for a track

        Args:
            track_id: Track ID

        Returns:
            Person ID or None if track not found
        """
        if track_id not in self._tracks:
            return None

        record = self._tracks[track_id]
        return record.person_id if not record.is_expired(self.cache_ttl) else None

    def needs_recognition(self, track_id: int) -> bool:
        """
        Check if track needs re-recognition

        Args:
            track_id: Track ID

        Returns:
            True if recognition needed
        """
        if track_id not in self._tracks:
            return True  # New track always needs recognition

        record = self._tracks[track_id]
        return record.needs_recognition(self.recognition_interval)

    def track_exists(self, track_id: int) -> bool:
        """Check if track exists and is not expired"""
        if track_id not in self._tracks:
            return False
        return not self._tracks[track_id].is_expired(self.cache_ttl)

    def get_track_record(self, track_id: int) -> Optional[TrackRecord]:
        """Get full track record"""
        if track_id not in self._tracks:
            return None
        record = self._tracks[track_id]
        if record.is_expired(self.cache_ttl):
            return None
        return record

    def cleanup(self, max_age: float = None):
        """
        Remove expired and stale tracks

        Args:
            max_age: Maximum age in seconds (default: cache_ttl * 2)
        """
        if max_age is None:
            max_age = self.cache_ttl * 2

        current_time = time.time()
        expired_ids = []

        for track_id, record in self._tracks.items():
            if (current_time - record.last_seen) > max_age:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            record = self._tracks[track_id]
            # Clean up person→track mapping
            if record.person_id != "unknown":
                self._remove_person_track_mapping(record.person_id, track_id)
            del self._tracks[track_id]

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired tracks")

    def match_by_embedding(
        self,
        embedding: np.ndarray,
        database,
        threshold: float = None,
        track_id: int = None
    ) -> Tuple[Optional[str], float]:
        """
        Match embedding against database and potentially update stats

        Args:
            embedding: Face embedding to match
            database: FaceDatabase instance
            threshold: Recognition threshold
            track_id: Track ID (for stats tracking)

        Returns:
            (person_id, confidence) or (None, 0.0)
        """
        if threshold is None:
            threshold = config.RECOGNITION_THRESHOLD

        results = database.search(embedding, threshold=threshold, top_k=1)

        if results:
            person_id, confidence = results[0]
            # Track stats
            if track_id is not None and track_id not in self._tracks:
                self.stats['occlusion_recoveries'] += 1
                logger.info(f"Occlusion recovery: track {track_id} matched to {person_id}")
            return person_id, confidence

        return "unknown", 0.0

    def get_stats(self) -> Dict:
        """Get manager statistics"""
        stats = self.stats.copy()
        stats['active_tracks'] = len(self._tracks)
        stats['cached_persons'] = len(self._person_tracks)
        return stats

    def _update_person_tracks_mapping(self, old_person: str, new_person: str, track_id: int):
        """Update person→tracks mapping when track's person changes"""
        # Remove from old person's tracks
        if old_person and old_person in self._person_tracks:
            if track_id in self._person_tracks[old_person]:
                self._person_tracks[old_person].remove(track_id)
            if not self._person_tracks[old_person]:
                del self._person_tracks[old_person]

        # Add to new person's tracks
        if new_person and new_person != "unknown":
            self._add_person_track_mapping(new_person, track_id)

    def _add_person_track_mapping(self, person_id: str, track_id: int):
        """Add track to person's track list"""
        if person_id not in self._person_tracks:
            self._person_tracks[person_id] = []
        if track_id not in self._person_tracks[person_id]:
            self._person_tracks[person_id].append(track_id)

    def _remove_person_track_mapping(self, person_id: str, track_id: int):
        """Remove track from person's track list"""
        if person_id in self._person_tracks:
            if track_id in self._person_tracks[person_id]:
                self._person_tracks[person_id].remove(track_id)
            if not self._person_tracks[person_id]:
                del self._person_tracks[person_id]

    def _evict_oldest_tracks(self, n: int = None):
        """
        Evict oldest N tracks based on last_seen

        Args:
            n: Number of tracks to evict (default: enough to get under max_cache_size)
        """
        if n is None:
            n = max(1, len(self._tracks) - self.max_cache_size + 1)

        # Sort by last_seen ascending (oldest first)
        sorted_tracks = sorted(
            self._tracks.items(),
            key=lambda x: x[1].last_seen
        )

        evict_count = 0
        for track_id, record in sorted_tracks[:n]:
            # Cleanup person mapping
            if record.person_id != "unknown":
                self._remove_person_track_mapping(record.person_id, track_id)
            del self._tracks[track_id]
            evict_count += 1

        logger.debug(f"Evicted {evict_count} oldest tracks from cache")
