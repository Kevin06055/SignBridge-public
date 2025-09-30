import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the SignBridge backend"""
    
    # Application
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    VERSION = '1.0.0'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Server
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # CORS
    CORS_ORIGINS = [
        'http://localhost:3000',  # React dev server
        'http://localhost:5173',  # Vite dev server
        'http://localhost:4173', 
         '*', # Vite preview
        os.environ.get('FRONTEND_URL', '')  # Production frontend URL
    ]
    
    # File Storage
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    OUTPUT_FOLDER = BASE_DIR / 'outputs'
    STATIC_FOLDER = BASE_DIR / 'static'
    
    # Text-to-Sign Configuration
    TEXT_TO_SIGN_IMAGES_DIR = BASE_DIR / 'TextToSignPipeline' / 'sign_images'
    TEXT_TO_SIGN_OUTPUT_DIR = BASE_DIR / 'TextToSignPipeline' / 'output_videos'
    MAX_TEXT_LENGTH = 1000
    DEFAULT_FPS = 1
    DEFAULT_FRAME_DURATION = 0.5
    VIDEO_RESOLUTION = (640, 480)
    SUPPORTED_VIDEO_FORMATS = ['mp4', 'avi']
    
    # Sign Detection Configuration  
    YOLO_MODEL_PATH = BASE_DIR.parent / 'fine_tuned.pt'  # Use the fine-tuned model
    YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get('YOLO_CONFIDENCE', 0.5))
    YOLO_FALLBACK_MODEL = BASE_DIR.parent / 'yolov8n.pt'  # Fallback model
    
    # Quiz Configuration
    QUIZ_SESSION_TIMEOUT = 1800  # 30 minutes
    MAX_QUIZ_OPTIONS = 4
    MIN_QUIZ_OPTIONS = 4
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / 'logs' / 'signbridge.log'
    
    # API Keys (for future authentication)
    API_KEYS = os.environ.get('API_KEYS', 'default-api-key').split(',')
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')
    
    # Firebase Configuration (for deployment)
    FIREBASE_PROJECT_ID = os.environ.get('FIREBASE_PROJECT_ID', '')
    FIREBASE_CREDENTIALS_PATH = os.environ.get('FIREBASE_CREDENTIALS_PATH', '')
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        for directory in [cls.UPLOAD_FOLDER, cls.OUTPUT_FOLDER, cls.STATIC_FOLDER, 
                         cls.TEXT_TO_SIGN_OUTPUT_DIR, cls.LOG_FILE.parent]:
            directory.mkdir(parents=True, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    LOG_LEVEL = 'WARNING'
    
    @classmethod
    def __init_subclass__(cls):
        if not cls.SECRET_KEY:
            raise ValueError("SECRET_KEY must be set in production")

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}