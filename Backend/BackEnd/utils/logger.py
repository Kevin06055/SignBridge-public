import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging(app):
    """Setup logging configuration for the Flask app"""
    
    # Ensure log directory exists
    log_dir = Path(app.config.get('LOG_FILE', 'logs/signbridge.log')).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set log level
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO').upper())
    
    # Configure Flask's logger
    app.logger.setLevel(log_level)
    
    # Remove default handlers
    if app.logger.handlers:
        app.logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    
    # File handler with rotation
    if not app.config.get('TESTING'):
        file_handler = RotatingFileHandler(
            app.config.get('LOG_FILE', 'logs/signbridge.log'),
            maxBytes=10240000,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        app.logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    app.logger.addHandler(console_handler)
    
    # Configure werkzeug logger (Flask's HTTP server)
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.WARNING if not app.debug else logging.INFO)
    
    app.logger.info("Logging configured successfully")