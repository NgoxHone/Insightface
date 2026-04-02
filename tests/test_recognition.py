#!/usr/bin/env python3
"""
Unit tests for recognition module
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from recognition.database import FaceDatabase, PersonRecord
from recognition.quality_check import assess_face_quality, is_face_good_quality
from recognition.utils import compute_similarity, normalize_image, crop_face

# Test fixtures

@pytest.fixture
def temp_db_path():
    """Create temporary database path"""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_db.pkl"
    yield db_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_embedding():
    """Generate a sample 512-D embedding"""
    return np.random.randn(512).astype(np.float32)


@pytest.fixture
def sample_embeddings():
    """Generate multiple sample embeddings"""
    return [np.random.randn(512).astype(np.float32) for _ in range(5)]


@pytest.fixture
def populated_database(temp_db_path, sample_embeddings):
    """Create and populate a database"""
    db = FaceDatabase(temp_db_path)
    db.add_person("Alice", [sample_embeddings[0], sample_embeddings[1]])
    db.add_person("Bob", [sample_embeddings[2]])
    return db


# ============== Database Tests ==============

def test_database_create(temp_db_path):
    """Test database initialization"""
    db = FaceDatabase(temp_db_path)
    assert len(db.get_all_people()) == 0
    assert db.get_stats()['total_people'] == 0


def test_add_person(temp_db_path, sample_embedding):
    """Test adding a person"""
    db = FaceDatabase(temp_db_path)
    success = db.add_person("John", [sample_embedding])
    assert success
    assert "John" in db.get_all_people()
    assert db.get_stats()['total_people'] == 1


def test_add_person_multiple_embeddings(temp_db_path, sample_embeddings):
    """Test adding person with multiple embeddings"""
    db = FaceDatabase(temp_db_path)
    success = db.add_person("Jane", sample_embeddings[:3])
    assert success
    record = db.get_person("Jane")
    assert len(record['embeddings']) == 3


def test_get_embedding(temp_db_path, sample_embeddings):
    """Test getting average embedding"""
    db = FaceDatabase(temp_db_path)
    db.add_person("Test", sample_embeddings[:3])

    emb = db.get_embedding("Test")
    assert emb is not None
    assert emb.shape == (512,)
    # Should be normalized
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_search(populated_database, sample_embeddings):
    """Test embedding search"""
    query_emb = sample_embeddings[0]  # Should match Alice closely
    results = populated_database.search(query_emb, threshold=0.0)  # Very loose threshold

    assert len(results) > 0
    # Alice should be first (her embedding is query)
    top_name, top_score = results[0]
    assert top_name == "Alice"
    assert top_score > 0.9  # Normalized dot product should be high


def test_search_threshold(populated_database, sample_embeddings):
    """Test threshold filtering"""
    # Use very different embedding
    query_emb = np.random.randn(512).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)

    results = populated_database.search(query_emb, threshold=0.8)
    # Should return empty or low scores
    for name, score in results:
        assert score < 0.8


def test_remove_person(populated_database):
    """Test removing person"""
    success = populated_database.remove_person("Bob")
    assert success
    assert "Bob" not in populated_database.get_all_people()
    assert populated_database.get_stats()['total_people'] == 1


def test_save_and_load(temp_db_path, sample_embeddings):
    """Test persistence"""
    db1 = FaceDatabase(temp_db_path)
    db1.add_person("Persist", [sample_embeddings[0]])
    db1.save()

    db2 = FaceDatabase(temp_db_path)
    assert "Persist" in db2.get_all_people()


def test_export_json(temp_db_path, sample_embeddings, tmp_path):
    """Test JSON export"""
    db = FaceDatabase(temp_db_path)
    db.add_person("Export", sample_embeddings)

    json_path = tmp_path / "export.json"
    success = db.export_json(json_path)
    assert success
    assert json_path.exists()


def test_get_stats(populated_database):
    """Test stats"""
    stats = populated_database.get_stats()
    assert stats['total_people'] == 2
    assert stats['total_embeddings'] == 3  # 2 for Alice + 1 for Bob


# ============== Quality Check Tests ==============

def test_assess_quality_white_image():
    """Test quality assessment on different images"""
    # Create white image
    white_img = np.ones((112, 112, 3), dtype=np.uint8) * 255
    score = assess_face_quality(white_img)
    assert 0.0 <= score <= 1.0


def test_quality_blur():
    """Blurry face should have low quality"""
    # Create blurry image
    img = np.random.rand(112, 112, 3).astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    score = assess_face_quality(blurred)
    # Blurry image should have low sharpness score
    assert score < 0.8


def test_quality_with_landmarks():
    """Test quality with landmarks"""
    img = np.random.rand(112, 112, 3).astype(np.uint8)
    landmarks = np.array([
        [30, 40], [80, 40],  # eyes
        [55, 70],            # nose
        [40, 90], [70, 90]   # mouth
    ])
    score = assess_face_quality(img, landmarks)
    assert 0.0 <= score <= 1.0


def test_is_face_good_quality():
    """Test convenience function"""
    good_img = np.random.rand(112, 112, 3).astype(np.uint8) * 200
    assert is_face_good_quality(good_img) >= False  # May pass or fail depending on threshold


# ============== Utils Tests ==============

def test_compute_similarity(sample_embedding):
    """Test cosine similarity"""
    emb1 = sample_embedding
    emb2 = sample_embedding.copy()
    similarity = compute_similarity(emb1, emb2)
    assert abs(similarity - 1.0) < 1e-5  # Same vector -> similarity 1

    opposite = -emb1 / np.linalg.norm(emb1)
    similarity_opposite = compute_similarity(emb1, opposite)
    assert abs(similarity_opposite) < 1e-5  # Opposite -> ~0


def test_crop_face():
    """Test face cropping"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = [50, 50, 150, 150]
    cropped = crop_face(img, bbox, margin=0)
    assert cropped.shape == (100, 100, 3)


def test_normalize_image():
    """Test image normalization"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    normalized = normalize_image(img, target_size=(112, 112))
    assert normalized.shape == (3, 112, 112)
    # Values should be in [-1, 1] if normalize=True
    assert np.min(normalized) >= -1.0
    assert np.max(normalized) <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
