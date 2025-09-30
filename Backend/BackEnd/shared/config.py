from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class SharedSettings(BaseSettings):
    # Application
    app_name: str = "Enhanced Multi-Pipeline Backend"
    version: str = "2.0.0"
    debug: bool = False
    
    # Server
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8080
    sign_detection_host: str = "0.0.0.0"
    sign_detection_port: int = 8001
    speech_to_sign_host: str = "0.0.0.0"
    speech_to_sign_port: int = 8002
    text_summarization_host: str = "0.0.0.0"
    text_summarization_port: int = 8003
    braille_conversion_host: str = "0.0.0.0"
    braille_conversion_port: int = 8004
    course_materials_host: str = "0.0.0.0"
    course_materials_port: int = 8005
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/pipelines_db"
    redis_url: str = "redis://localhost:6379"
    
    # Security
    secret_key: str = "your-super-secret-key-change-in-production"
    api_keys: List[str] = ["default-api-key"]
    cors_origins: List[str] = ["*"]
    
    # Google Cloud
    google_application_credentials: Optional[str] = None
    gcp_project_id: str = "your-project-id"
    
    # File Storage
    upload_path: str = "/tmp/uploads"
    max_file_size_mb: int = 50
    
    # Sign Detection (YOLO)
    sign_detection_model_path: str = "models/SignConv.pt"
    yolo_confidence_threshold: float = 0.8
    yolo_max_detections: int = 10
    
    # Sign Language Videos/GIFs
    gifs_directory: str = "assets/sign_gifs"
    video_cdn_base_url: str = "https://cdn.example.com/signs"
    default_frame_duration: float = 1.5
    supported_video_formats: List[str] = ["mp4", "avi", "mov", "gif"]
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = SharedSettings()
