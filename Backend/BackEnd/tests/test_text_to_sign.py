"""
Tests for text-to-sign blueprint
"""
import json
import time
import pytest

def test_text_to_sign_health(client):
    """Test text-to-sign health endpoint"""
    response = client.get('/api/v1/text-to-sign/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['service'] == 'text-to-sign'
    assert 'status' in data

def test_available_letters(client):
    """Test available letters endpoint"""
    response = client.get('/api/v1/text-to-sign/available-letters')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'data' in data
    assert 'available_letters' in data['data']
    assert 'count' in data['data']
    
    # Should have our test letters A, B, C
    letters = data['data']['available_letters']
    assert 'A' in letters
    assert 'B' in letters
    assert 'C' in letters

def test_convert_text_to_sign_valid_input(client):
    """Test text-to-sign conversion with valid input"""
    request_data = {
        'text': 'ABC',
        'options': {
            'frame_duration': 1.0,
            'video_fps': 1,
            'filename': 'test_video.mp4'
        }
    }
    
    response = client.post('/api/v1/text-to-sign/convert',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'data' in data
    assert 'task_id' in data['data']
    assert 'status_url' in data['data']
    assert data['data']['status'] == 'processing'

def test_convert_text_to_sign_no_text(client):
    """Test text-to-sign conversion with no text"""
    request_data = {}
    
    response = client.post('/api/v1/text-to-sign/convert',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_convert_text_to_sign_empty_text(client):
    """Test text-to-sign conversion with empty text"""
    request_data = {'text': '   '}
    
    response = client.post('/api/v1/text-to-sign/convert',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_convert_text_to_sign_too_long(client):
    """Test text-to-sign conversion with text too long"""
    request_data = {'text': 'A' * 2000}  # Exceeds MAX_TEXT_LENGTH
    
    response = client.post('/api/v1/text-to-sign/convert',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'too long' in data['error'].lower()

def test_task_status_not_found(client):
    """Test task status for non-existent task"""
    response = client.get('/api/v1/text-to-sign/status/nonexistent-task-id')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert 'error' in data

def test_download_video_not_found(client):
    """Test video download for non-existent task"""
    response = client.get('/api/v1/text-to-sign/download/nonexistent-task-id')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert 'error' in data