"""
Tests for the main Flask application
"""
import json
import pytest

def test_health_endpoint(client):
    """Test health endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data
    assert 'version' in data

def test_api_info_endpoint(client):
    """Test API info endpoint"""
    response = client.get('/api')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'service' in data
    assert 'version' in data
    assert 'endpoints' in data
    assert 'text_to_sign' in data['endpoints']
    assert 'sign_detection' in data['endpoints']

def test_cors_headers(client):
    """Test CORS headers are present"""
    response = client.get('/health')
    assert 'Access-Control-Allow-Origin' in response.headers
    assert 'Access-Control-Allow-Methods' in response.headers
    assert 'Access-Control-Allow-Headers' in response.headers

def test_404_error_handler(client):
    """Test 404 error handler"""
    response = client.get('/nonexistent')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert 'error' in data
    assert data['status'] == 404

def test_options_method(client):
    """Test OPTIONS method for CORS preflight"""
    response = client.options('/api')
    assert response.status_code in [200, 204]  # Different Flask versions handle this differently