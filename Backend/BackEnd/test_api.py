#!/usr/bin/env python3
"""
Test script for SignBridge backend
"""

import requests
import time
import json

def test_api():
    base_url = 'http://localhost:5000'

    print("Testing SignBridge Backend API...")

    # Test health endpoint
    try:
        response = requests.get(f'{base_url}/health')
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # Test available letters
    try:
        response = requests.get(f'{base_url}/api/v1/text-to-sign/available-letters')
        print(f"Available letters: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Available letters failed: {e}")
        return

    # Test text-to-sign conversion
    try:
        response = requests.post(
            f'{base_url}/api/v1/text-to-sign/convert',
            json={'text': 'A'},
            headers={'Content-Type': 'application/json'}
        )
        print(f"Text-to-sign convert: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        if 'data' in data and 'task_id' in data['data']:
            task_id = data['data']['task_id']
            print(f"Task ID: {task_id}")

            # Wait and check status
            time.sleep(5)
            status_response = requests.get(f'{base_url}/api/v1/text-to-sign/status/{task_id}')
            print(f"Status check: {status_response.status_code}")
            print(f"Status: {status_response.json()}")

    except Exception as e:
        print(f"Text-to-sign conversion failed: {e}")

if __name__ == '__main__':
    test_api()