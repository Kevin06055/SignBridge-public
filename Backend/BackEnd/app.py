#!/usr/bin/env python3
"""
SignBridge Unified Flask Backend
A production-ready Flask backend that integrates text-to-sign conversion and sign detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime

from config import Config
from blueprints.text_to_sign import text_to_sign_bp
from blueprints.sign_detection import sign_detection_bp
from blueprints.speech_to_sign import speech_to_sign_bp
from blueprints.braille_conversion import braille_conversion_bp
from blueprints.course_material import course_material_bp
from utils.logger import setup_logging

def create_app(config_class=Config):
    """Application factory pattern"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Setup logging
    setup_logging(app)
    
    # Enable CORS for frontend integration
    CORS(app, resources={
        r"/api/*": {
            "origins": app.config.get('CORS_ORIGINS', [
                'http://localhost:3000', 
                'http://localhost:5173',
                'http://localhost:8080',
                'http://localhost:8081',
                'http://127.0.0.1:3000',
                'http://127.0.0.1:5173',
                'http://127.0.0.1:8080',
                'http://127.0.0.1:8081'
            ]),
            "allow_headers": ["Content-Type", "Authorization", "Accept", "X-API-Key"],
            "methods": ["GET", "POST", "OPTIONS"],
            "expose_headers": ["Content-Disposition"]
        }
    })
    
    # Add CORS headers to all responses
    @app.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        allowed_origins = [
            'http://localhost:3000', 'http://localhost:5173',
            'http://localhost:8080', 'http://localhost:8081',
            'http://127.0.0.1:3000', 'http://127.0.0.1:5173',
            'http://127.0.0.1:8080', 'http://127.0.0.1:8081'
        ]
        
        if origin in allowed_origins:
            response.headers.add('Access-Control-Allow-Origin', origin)
        else:
            response.headers.add('Access-Control-Allow-Origin', '*')
            
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,X-API-Key')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        response.headers.add('Access-Control-Expose-Headers', 'Content-Disposition')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    # Register blueprints
    app.register_blueprint(text_to_sign_bp, url_prefix='/api/v1/text-to-sign')
    app.register_blueprint(sign_detection_bp, url_prefix='/api/v1/sign-detection')
    app.register_blueprint(speech_to_sign_bp, url_prefix='/api/v1/speech-to-sign')
    app.register_blueprint(braille_conversion_bp, url_prefix='/api/v1/braille-conversion')
    app.register_blueprint(course_material_bp, url_prefix='/api/v1/course-materials')
    
    # Health check endpoint
    @app.route('/health')
    @app.route('/api/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': app.config.get('VERSION', '1.0.0'),
            'service': 'SignBridge Backend'
        })
    
    # Root API endpoint
    @app.route('/api')
    def api_info():
        """API information endpoint"""
        return jsonify({
            'service': 'SignBridge Unified Backend',
            'version': app.config.get('VERSION', '1.0.0'),
            'endpoints': {
                'text_to_sign': '/api/v1/text-to-sign',
                'sign_detection': '/api/v1/sign-detection',
                'speech_to_sign': '/api/v1/speech-to-sign',
                'braille_conversion': '/api/v1/braille-conversion',
                'course_materials': '/api/v1/course-materials',
                'health': '/health'
            },
            'documentation': 'https://github.com/your-org/signbridge'
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found', 'status': 404}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error', 'status': 500}), 500
    
    return app

# Create the app instance
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.logger.info(f"Starting SignBridge Backend on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)