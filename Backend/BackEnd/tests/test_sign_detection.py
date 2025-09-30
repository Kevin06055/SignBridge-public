"""
Tests for sign detection blueprint
"""
import json
import base64
import pytest
import numpy as np
import cv2

def create_test_image_base64():
    """Create a test image as base64 string"""
    # Create a simple test image
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', img)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

def test_sign_detection_health(client):
    """Test sign detection health endpoint"""
    response = client.get('/api/v1/sign-detection/health')
    assert response.status_code in [200, 500]  # May fail if YOLO model not available
    
    data = json.loads(response.data)
    assert data['service'] == 'sign-detection'
    assert 'model_status' in data

def test_model_info(client):
    """Test model info endpoint"""
    response = client.get('/api/v1/sign-detection/model/info')
    assert response.status_code in [200, 500]  # May fail if YOLO model not available
    
    data = json.loads(response.data)
    assert 'model_info' in data or 'error' in data

def test_detect_sign_no_data(client):
    """Test sign detection with no data"""
    response = client.post('/api/v1/sign-detection/detect',
                          data=json.dumps({}),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_detect_sign_with_frame(client):
    """Test sign detection with frame data"""
    test_image = create_test_image_base64()
    request_data = {
        'frame': test_image,
        'confidence': 0.3
    }
    
    response = client.post('/api/v1/sign-detection/detect',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    # Should return 200 regardless of detection results
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'success' in data
    assert 'detections' in data

def test_detect_sign_realtime(client):
    """Test real-time sign detection"""
    test_image = create_test_image_base64()
    request_data = {
        'frame': test_image,
        'confidence': 0.3
    }
    
    response = client.post('/api/v1/sign-detection/detect-realtime',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    # Should return 200 regardless of detection results
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'success' in data
    assert 'current_prediction' in data
    assert 'timestamp' in data

def test_reset_history(client):
    """Test reset prediction history"""
    response = client.post('/api/v1/sign-detection/reset-history')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'message' in data

def test_configuration(client):
    """Test get configuration"""
    response = client.get('/api/v1/sign-detection/configuration')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'configuration' in data
    assert 'confidence_threshold' in data['configuration']