"""
Firebase Functions entry point for SignBridge Backend
"""
import os
from functions_framework import create_app
from app import app

# Firebase Functions requires a specific naming convention
signbridge_backend = app

# For local testing
if __name__ == "__main__":
    app.run(debug=True)