import pytest
import os
import tempfile
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from config import Config

class TestConfig(Config):
    """Test configuration"""
    TESTING = True
    DEBUG = True
    SECRET_KEY = 'test-secret-key'
    TEXT_TO_SIGN_IMAGES_DIR = Path(__file__).parent / 'test_data' / 'sign_images'
    TEXT_TO_SIGN_OUTPUT_DIR = Path(__file__).parent / 'test_data' / 'output'
    YOLO_MODEL_PATH = Path(__file__).parent.parent.parent / 'yolov8n.pt'  # Use fallback model for tests
    YOLO_CONFIDENCE_THRESHOLD = 0.3

@pytest.fixture
def app():
    """Create test app"""
    app = create_app(TestConfig)
    
    # Ensure test directories exist
    TestConfig.TEXT_TO_SIGN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TestConfig.TEXT_TO_SIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create some dummy sign images for testing
    import cv2
    import numpy as np
    
    for letter in ['A', 'B', 'C']:
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img, letter, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
        cv2.imwrite(str(TestConfig.TEXT_TO_SIGN_IMAGES_DIR / f'{letter}.png'), img)
    
    yield app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create test CLI runner"""
    return app.test_cli_runner()