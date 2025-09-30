#!/usr/bin/env python3
"""
SignBridge Backend Startup Script
Simple script to run the SignBridge unified Flask backend
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app import app
from config import Config

def main():
    """Main entry point"""
    # Ensure required directories exist
    Config.ensure_directories()
    
    # Get configuration from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Starting SignBridge Backend")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"   Endpoints available at:")
    print(f"     - Health: http://{host}:{port}/health")
    print(f"     - Text-to-Sign: http://{host}:{port}/api/v1/text-to-sign")
    print(f"     - Sign Detection: http://{host}:{port}/api/v1/sign-detection")
    print(f"     - Speech-to-Sign: http://{host}:{port}/api/v1/speech-to-sign")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 SignBridge Backend stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()