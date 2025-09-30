"""
Tests for speech-to-sign blueprint
"""
import json
import time
import pytest

def test_speech_to_sign_health(client):
    """Test speech-to-sign health endpoint"""
    response = client.get('/api/v1/speech-to-sign/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['service'] == 'speech-to-sign'
    assert data['status'] == 'healthy'
    assert 'available_letters' in data
    assert 'processor_available' in data

def test_speech_available_letters(client):
    """Test speech available letters endpoint"""
    response = client.get('/api/v1/speech-to-sign/available-letters')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'data' in data
    assert 'available_letters' in data['data']
    assert data['data']['service'] == 'speech-to-sign'

def test_convert_speech_to_sign_valid_input(client):
    """Test speech-to-sign conversion with valid input"""
    request_data = {
        'text': 'HELLO ABC',
        'frame_duration': 1.0,
        'video_fps': 1,
        'filename': 'speech_test_video.mp4'
    }
    
    response = client.post('/api/v1/speech-to-sign/convert',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'data' in data
    assert 'task_id' in data['data']
    assert data['data']['status'] == 'processing'

def test_convert_speech_to_sign_no_text(client):
    """Test speech-to-sign conversion with no text"""
    request_data = {}
    
    response = client.post('/api/v1/speech-to-sign/convert',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_convert_speech_to_sign_empty_text(client):
    """Test speech-to-sign conversion with empty text"""
    request_data = {'text': '   '}
    
    response = client.post('/api/v1/speech-to-sign/convert',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_speech_preview_conversion(client):
    """Test speech preview functionality"""
    request_data = {'text': 'HELLO WORLD ABC'}
    
    response = client.post('/api/v1/speech-to-sign/preview',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'data' in data
    assert 'letters' in data['data']
    assert 'coverage_percentage' in data['data']

def test_speech_task_status_not_found(client):
    """Test speech task status for non-existent task"""
    response = client.get('/api/v1/speech-to-sign/status/nonexistent-speech-task-id')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert data['success'] is False
    assert 'error' in data

def test_speech_download_video_not_found(client):
    """Test speech video download for non-existent task"""
    response = client.get('/api/v1/speech-to-sign/download/nonexistent-speech-task-id')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert data['success'] is False
    assert 'error' in data

def test_process_text_alternative_endpoint(client):
    """Test alternative process-text endpoint"""
    request_data = {
        'text': 'TEST ABC',
        'filename': 'alt_test_video.mp4'
    }
    
    response = client.post('/api/v1/speech-to-sign/process-text',
                          data=json.dumps(request_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'data' in data