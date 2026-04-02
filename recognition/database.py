"""
Face Database: File-based storage for face embeddings using pickle/JSON
"""
import logging
import pickle
import json
import cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from config import config

logger = logging.getLogger(__name__)


@dataclass
class PersonRecord:
    """Data class for a person's face records"""
    name: str
    embeddings: List[np.ndarray]
    metadata: Dict[str, any]

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'embedding_count': len(self.embeddings),
                'updated_at': datetime.now().isoformat()
            }
        else:
            self.metadata['embedding_count'] = len(self.embeddings)
            self.metadata['updated_at'] = datetime.now().isoformat()


class FaceDatabase:
    """
    Face embedding database with pickle/JSON persistence

    Storage format:
    {
        "person_name": {
            "embeddings": [[512-d], [512-d], ...],
            "metadata": {...}
        }
    }
    """

    def __init__(self, db_path: Optional[Path] = None, json_path: Optional[Path] = None, auto_reload: bool = True):
        """
        Initialize database

        Args:
            db_path: Path to pickle database file
            json_path: Path to JSON backup file
            auto_reload: Automatically reload if file changes
        """
        self.db_path = db_path or config.DATABASE_PATH
        self.json_path = json_path or config.DATABASE_JSON_PATH
        self._data: Dict[str, Dict] = {}
        self._loaded = False
        self._auto_reload = auto_reload
        self._last_modified = 0
        self.load()

    def load(self, force: bool = False) -> bool:
        """
        Load database from disk

        Args:
            force: Force reload even if file hasn't changed

        Returns:
            True if successful
        """
        try:
            # Check if file has changed (for auto-reload)
            if not force and self._auto_reload and self.db_path.exists():
                current_modified = self.db_path.stat().st_mtime
                if self._loaded and current_modified <= self._last_modified:
                    return True  # No changes, keep current data
                self._last_modified = current_modified

            if self.db_path.exists():
                with open(self.db_path, 'rb') as f:
                    self._data = pickle.load(f)
                logger.debug(f"Loaded database with {len(self._data)} people from {self.db_path}")
                self._loaded = True
                return True
            else:
                logger.info("No existing database found, starting fresh")
                self._data = {}
                self._loaded = True
                self._last_modified = 0
                return True
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            self._data = {}
            return False

    def save(self, use_json: bool = False) -> bool:
        """
        Save database to disk

        Args:
            use_json: Save as JSON (human-readable but slower) instead of pickle

        Returns:
            True if successful
        """
        try:
            if use_json:
                # Convert numpy arrays to lists for JSON
                json_data = {}
                for name, record in self._data.items():
                    json_data[name] = {
                        'embeddings': [emb.tolist() for emb in record['embeddings']],
                        'metadata': record['metadata']
                    }
                with open(self.json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                logger.info(f"Saved database to JSON: {self.json_path}")
                # Update modified time
                if self.json_path.exists():
                    self._last_modified = self.json_path.stat().st_mtime
            else:
                with open(self.db_path, 'wb') as f:
                    pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved database to pickle: {self.db_path}")
                # Update modified time
                if self.db_path.exists():
                    self._last_modified = self.db_path.stat().st_mtime
            return True
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            return False

    def add_person(
        self,
        name: str,
        embeddings: List[np.ndarray],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add or update a person in database

        Args:
            name: Person's name
            embeddings: List of embedding vectors (512-D)
            metadata: Optional additional metadata

        Returns:
            True if successful
        """
        try:
            if not embeddings:
                logger.warning(f"No embeddings provided for {name}")
                return False

            # Validate embeddings
            valid_embeddings = []
            for emb in embeddings:
                if emb is not None and emb.shape == (512,):
                    valid_embeddings.append(emb.astype(np.float32))
                else:
                    logger.warning(f"Invalid embedding shape: {emb.shape if emb is not None else 'None'}")

            if not valid_embeddings:
                logger.warning(f"No valid embeddings for {name}")
                return False

            # If updating existing person, merge embeddings (up to limit)
            existing_count = 0
            if name in self._data:
                existing_embeddings = self._data[name]['embeddings']
                existing_count = len(existing_embeddings)
                # Keep some of old embeddings + new ones
                keep_existing = min(existing_count, config.MAX_EMBEDDINGS_PER_PERSON // 2)
                valid_embeddings = existing_embeddings[:keep_existing] + valid_embeddings
                logger.info(f"Updating {name}: {existing_count} old + {len(embeddings)} new embeddings")

            # Limit total embeddings per person
            if len(valid_embeddings) > config.MAX_EMBEDDINGS_PER_PERSON:
                valid_embeddings = valid_embeddings[:config.MAX_EMBEDDINGS_PER_PERSON]
                logger.info(f"Truncated embeddings for {name} to {config.MAX_EMBEDDINGS_PER_PERSON}")

            # Create record
            record = {
                'embeddings': valid_embeddings,
                'metadata': {
                    **(metadata or {}),
                    'created_at': datetime.now().isoformat() if name not in self._data else self._data[name]['metadata'].get('created_at'),
                    'updated_at': datetime.now().isoformat(),
                    'embedding_count': len(valid_embeddings)
                }
            }

            self._data[name] = record
            self.save()  # Auto-save
            logger.info(f"Added/updated {name} with {len(valid_embeddings)} embeddings")
            return True

        except Exception as e:
            logger.error(f"Failed to add person {name}: {e}")
            return False

    def remove_person(self, name: str) -> bool:
        """
        Remove a person from database

        Args:
            name: Person's name to remove

        Returns:
            True if successful
        """
        try:
            if name in self._data:
                del self._data[name]
                self.save()
                logger.info(f"Removed {name} from database")
                return True
            else:
                logger.warning(f"Person {name} not found in database")
                return False
        except Exception as e:
            logger.error(f"Failed to remove person {name}: {e}")
            return False

    def get_person(self, name: str) -> Optional[Dict]:
        """Get person record"""
        return self._data.get(name)

    def _ensure_reload(self):
        """Check and reload if database file changed"""
        if self._auto_reload:
            self.load(force=False)

    def get_all_people(self) -> List[str]:
        """Get list of all person names"""
        self._ensure_reload()
        return list(self._data.keys())

    def get_embedding(
        self,
        name: str,
        use_average: bool = None
    ) -> Optional[np.ndarray]:
        """
        Get average embedding for a person

        Args:
            name: Person's name
            use_average: Override config.USE_AVERAGE_EMBEDDING

        Returns:
            Average embedding (512-D) or None
        """
        record = self._data.get(name)
        if not record:
            return None

        embeddings = record['embeddings']
        if not embeddings:
            return None

        if use_average is None:
            use_average = config.USE_AVERAGE_EMBEDDING

        if use_average and len(embeddings) > 1:
            # Return normalized average
            avg_emb = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg_emb)
            if norm > 0:
                return avg_emb / norm
            return avg_emb
        else:
            # Return first embedding (normalized)
            emb = embeddings[0]
            norm = np.linalg.norm(emb)
            if norm > 0:
                return emb / norm
            return emb

    def search(
        self,
        query_embedding: np.ndarray,
        threshold: float = None,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for matching faces in database

        Args:
            query_embedding: 512-D embedding to search
            threshold: Minimum similarity threshold
            top_k: Maximum number of results

        Returns:
            List of (name, similarity) tuples, sorted by similarity descending
        """
        if threshold is None:
            threshold = config.RECOGNITION_THRESHOLD

        if query_embedding is None:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        results = []

        for name, record in self._data.items():
            embeddings = record['embeddings']
            if not embeddings:
                continue

            # Compare with all embeddings for this person
            similarities = []
            for emb in embeddings:
                emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
                similarity = np.dot(query_norm, emb_norm)
                similarities.append(similarity)

            # Use max or average similarity
            if config.USE_AVERAGE_EMBEDDING and len(similarities) > 1:
                similarity = np.mean(similarities)
            else:
                similarity = max(similarities)

            if similarity >= threshold:
                results.append((name, float(similarity)))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def get_stats(self) -> Dict:
        """Get database statistics"""
        total_embeddings = sum(len(record['embeddings']) for record in self._data.values())
        return {
            'total_people': len(self._data),
            'total_embeddings': total_embeddings,
            'average_embeddings_per_person': total_embeddings / len(self._data) if self._data else 0,
            'people': list(self._data.keys())
        }

    def clear(self):
        """Clear all data"""
        self._data = {}
        self.save()
        logger.info("Database cleared")

    def export_json(self, path: Optional[Path] = None) -> bool:
        """Export database to JSON format"""
        export_path = path or self.json_path
        return self.save(use_json=True)

    def import_json(self, path: Optional[Path] = None) -> bool:
        """Import database from JSON format"""
        import_path = path or self.json_path
        try:
            with open(import_path, 'r') as f:
                json_data = json.load(f)

            self._data = {}
            for name, record in json_data.items():
                self._data[name] = {
                    'embeddings': [np.array(emb, dtype=np.float32) for emb in record['embeddings']],
                    'metadata': record['metadata']
                }

            self.save()
            logger.info(f"Imported database from {import_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to import database: {e}")
            return False

    def save_person_images(self, name: str, images: List[np.ndarray]) -> bool:
        """
        Save raw face images for a person to data/raw/

        Args:
            name: Person name
            images: List of face images (BGR)

        Returns:
            True if successful
        """
        try:
            from config import config
            raw_dir = config.DATA_DIR / "raw" / name
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Get existing count to avoid overwriting
            existing = list(raw_dir.glob("*.jpg"))
            start_idx = len(existing)

            for i, img in enumerate(images):
                filename = raw_dir / f"{start_idx + i + 1:04d}.jpg"
                cv2.imwrite(str(filename), img)

            logger.info(f"Saved {len(images)} raw images for {name} to {raw_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to save images for {name}: {e}")
            return False
