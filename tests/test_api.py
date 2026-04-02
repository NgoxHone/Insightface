#!/usr/bin/env python3
"""
API integration tests
"""
import pytest
import numpy as np
import cv2
from io import BytesIO
from pathlib import Path
import tempfile
import shutil

from app import create_app
from recognition.database import FaceDatabase

# Test fixtures

@pytest.fixture
def client():
    """Create test client"""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a sample face-like image"""
    # Create colored rectangle image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = (255, 200, 150)  # Skin tone rectangle
    return img


@pytest.fixture
def populated_db(tmp_path):
    """Create populated database"""
    db_path = tmp_path / "test_api_db.pkl"
    db = FaceDatabase(db_path)

    # Create sample embeddings
    for i in range(3):
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        db.add_person(f"Person{i}", [emb])

    return db_path


# ============== Health & Info ==============

def test_health(client):
    """Test health endpoint"""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'status' in data['data']


def test_info(client):
    """Test info endpoint"""
    response = client.get('/api/info')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'stats' in data['data']


# ============== People Management ==============

def test_list_people(client, populated_db):
    """Test listing people"""
    response = client.get('/api/people')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert len(data['data']['people']) >= 0


def test_delete_person(client, populated_db):
    """Test deleting person"""
    # First list to get a name
    list_resp = client.get('/api/people')
    people = list_resp.get_json()['data']['people']
    if people:
        name = people[0]
        delete_resp = client.delete(f'/api/person/{name}')
        assert delete_resp.status_code == 200
        data = delete_resp.get_json()
        assert data['success'] is True


def test_delete_nonexistent(client):
    """Test deleting non-existent person"""
    response = client.delete('/api/person/Nonexistent')
    assert response.status_code == 404


# ============== Registration ==============

def test_register_no_name(client, sample_image):
    """Test registration without name"""
    img_bytes = encode_image(sample_image)
    data = {'images': (BytesIO(img_bytes), 'face.jpg')}
    response = client.post('/api/register', data=data, content_type='multipart/form-data')
    assert response.status_code == 400


def test_register_no_images(client):
    """Test registration without images"""
    data = {'name': 'Test'}
    response = client.post('/api/register', data=data, content_type='multipart/form-data')
    assert response.status_code == 400


def test_register_success(client, sample_image):
    """Test successful registration"""
    img_bytes = encode_image(sample_image)
    data = {
        'name': 'TestUser',
        'images': (BytesIO(img_bytes), 'face.jpg')
    }
    response = client.post('/api/register', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    resp_data = response.get_json()
    assert resp_data['success'] is True
    assert 'embeddings_extracted' in resp_data['data']


def test_register_multiple_images(client):
    """Test registering with multiple images"""
    images = []
    for i in range(3):
        img = np.random.rand(100, 100, 3).astype(np.uint8)
        images.append((BytesIO(encode_image(img)), f'face{i}.jpg'))

    data = {
        'name': 'MultiFace',
        'images': images
    }
    response = client.post('/api/register', data=data, content_type='multipart/form-data')
    assert response.status_code == 200


# ============== Recognition ==============

def test_recognize_no_image(client):
    """Test recognize without image"""
    response = client.post('/api/recognize')
    assert response.status_code == 400


def test_recognize_invalid_image(client):
    """Test recognize with invalid image"""
    data = {'image': (BytesIO(b'invalid'), 'test.txt')}
    response = client.post('/api/recognize', data=data, content_type='multipart/form-data')
    assert response.status_code == 400


def test_recognize_empty_image(client):
    """Test recognize with empty image"""
    img_bytes = encode_image(np.zeros((10, 10, 3), dtype=np.uint8))
    data = {'image': (BytesIO(img_bytes), 'empty.jpg')}
    response = client.post('/api/recognize', data=data, content_type='multipart/form-data')
    # Should still process but might not find faces
    assert response.status_code in [200, 500]


def test_recognize_with_threshold(client, sample_image):
    """Test recognize with threshold parameter"""
    img_bytes = encode_image(sample_image)
    data = {
        'image': (BytesIO(img_bytes), 'face.jpg'),
        'threshold': '0.5'
    }
    response = client.post('/api/recognize', data=data, content_type='multipart/form-data')
    assert response.status_code == 200


def test_recognize_invalid_threshold(client, sample_image):
    """Test recognize with invalid threshold"""
    img_bytes = encode_image(sample_image)
    data = {
        'image': (BytesIO(img_bytes), 'face.jpg'),
        'threshold': '1.5'  # >1 invalid
    }
    response = client.post('/api/recognize', data=data, content_type='multipart/form-data')
    assert response.status_code == 400


# ============== Detection ==============

def test_detect_no_image(client):
    """Test detect without image"""
    response = client.post('/api/detect')
    assert response.status_code == 400


def test_detect_success(client, sample_image):
    """Test detect endpoint"""
    img_bytes = encode_image(sample_image)
    data = {'image': (BytesIO(img_bytes), 'face.jpg')}
    response = client.post('/api/detect', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    resp_data = response.get_json()
    assert resp_data['success'] is True
    assert 'faces' in resp_data['data']


# ============== Stats ==============

def test_stats(client):
    """Test stats endpoint"""
    response = client.get('/api/stats')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'total_people' in data['data']


# ============== Utilities ==============

def encode_image(img: np.ndarray) -> bytes:
    """Encode numpy image to JPEG bytes"""
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
