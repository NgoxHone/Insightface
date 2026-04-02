"""
Pytest configuration and fixtures
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from config import config


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test session"""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture(scope="session")
def temp_data_dir(temp_dir):
    """Create temporary data directory"""
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture(scope="session")
def temp_uploads_dir(temp_dir):
    """Create temporary uploads directory"""
    uploads_dir = temp_dir / "uploads"
    uploads_dir.mkdir(parents=True)
    return uploads_dir


# Override config paths for testing
@pytest.fixture(autouse=True)
def override_config(temp_data_dir, temp_uploads_dir, monkeypatch):
    """Override config paths to use temp directories during tests"""
    monkeypatch.setattr(config, 'DATA_DIR', temp_data_dir)
    monkeypatch.setattr(config, 'UPLOAD_DIR', temp_uploads_dir)
    monkeypatch.setattr(config, 'DATABASE_PATH', temp_data_dir / "test_db.pkl")
    monkeypatch.setattr(config, 'LOG_FILE', temp_dir / "test.log")
    yield
