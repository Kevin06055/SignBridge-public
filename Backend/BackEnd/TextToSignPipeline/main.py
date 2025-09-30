from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import uvicorn
from typing import Optional, List, Dict, Any
import os
from datetime import datetime, timedelta
import cv2
import numpy as np
from PIL import Image
import tempfile
import threading
import time
import json
import random
import base64
import io
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# FastAPI app instance
app = FastAPI(
    title="Text-to-Sign Pipeline",
    description="Convert text to sign language video output",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    SIGN_IMAGES_DIR = "sign_images"  # Directory containing A.png, B.png, etc.
    OUTPUT_DIR = "output_videos"
    MAX_TEXT_LENGTH = 1000
    DEFAULT_FPS = 1  # Frames per second for video
    DEFAULT_FRAME_DURATION = 0.5  # Seconds per letter
    VIDEO_RESOLUTION = (640, 480)
    SUPPORTED_FORMATS = ['mp4', 'avi']
    # Quiz configuration
    QUIZ_SESSION_TIMEOUT = 1800  # 30 minutes
    MAX_QUIZ_OPTIONS = 4
    MIN_QUIZ_OPTIONS = 4

# Request/Response models
class TextToSignRequest(BaseModel):
    text: str
    output_format: str = "mp4"
    fps: float = 1.0
    frame_duration: float = 0.5

class TextToSignResponse(BaseModel):
    task_id: str
    status: str
    message: str
    video_url: Optional[str] = None
    processing_details: Optional[Dict[str, Any]] = None

class QuizRequest(BaseModel):
    text: str
    difficulty: str = "medium"  # easy, medium, hard
    num_questions: int = 5

class QuizResponse(BaseModel):
    quiz_id: str
    questions: List[Dict[str, Any]]
    expires_at: str

# Processing status tracker
class ProcessingStatus:
    def __init__(self):
        self.status = {}
        self.lock = threading.Lock()
    
    def start_new_task(self, task_id):
        with self.lock:
            self.status[task_id] = {
                'progress': 0,
                'status': 'processing',
                'message': 'Initializing...',
                'start_time': datetime.now().isoformat(),
                'completed': False,
                'result': None
            }
        return task_id
    
    def update_progress(self, task_id, progress, message=None):
        with self.lock:
            if task_id in self.status:
                self.status[task_id]['progress'] = progress
                if message:
                    self.status[task_id]['message'] = message
    
    def complete_task(self, task_id, result=None):
        with self.lock:
            if task_id in self.status:
                self.status[task_id]['progress'] = 100
                self.status[task_id]['status'] = 'completed'
                self.status[task_id]['completed'] = True
                self.status[task_id]['end_time'] = datetime.now().isoformat()
                if result:
                    self.status[task_id]['result'] = result
    
    def get_status(self, task_id):
        with self.lock:
            return self.status.get(task_id, None)
    
    def cleanup_old_tasks(self):
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=24)
            tasks_to_remove = []
            for task_id, task_data in self.status.items():
                task_time = datetime.fromisoformat(task_data['start_time'])
                if task_time < cutoff:
                    tasks_to_remove.append(task_id)
            for task_id in tasks_to_remove:
                del self.status[task_id]

# Global processing status tracker
processing_status = ProcessingStatus()

# Text-to-Sign Conversion Core Logic
class SignLanguageConverter:
    def __init__(self, config: Config):
        self.config = config
        self.sign_images_path = os.path.join(os.path.dirname(__file__), config.SIGN_IMAGES_DIR)
        self.output_path = os.path.join(os.path.dirname(__file__), config.OUTPUT_DIR)
        
        # Ensure directories exist
        os.makedirs(self.sign_images_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        
        logger.info(f"Sign images path: {self.sign_images_path}")
        logger.info(f"Output path: {self.output_path}")
    
    def text_to_video(self, text: str, task_id: str, fps: float = None, frame_duration: float = None) -> str:
        """Convert text to sign language video"""
        try:
            # Use defaults if not specified
            fps = fps or self.config.DEFAULT_FPS
            frame_duration = frame_duration or self.config.DEFAULT_FRAME_DURATION
            
            processing_status.update_progress(task_id, 10, "Processing text input")
            
            # Clean and validate text
            text = re.sub(r'[^a-zA-Z\s]', '', text.upper())
            text = ' '.join(text.split())  # Remove extra spaces
            
            if len(text) > self.config.MAX_TEXT_LENGTH:
                text = text[:self.config.MAX_TEXT_LENGTH]
            
            if not text:
                raise ValueError("No valid characters found in text")
            
            processing_status.update_progress(task_id, 25, "Loading sign images")
            
            # Get available sign images
            available_signs = self._get_available_signs()
            
            # Create frames for video
            frames = []
            total_chars = len([c for c in text if c.isalpha()])
            processed_chars = 0
            
            processing_status.update_progress(task_id, 40, "Generating video frames")
            
            for char in text:
                if char.isalpha():
                    sign_path = os.path.join(self.sign_images_path, f"{char}.png")
                    
                    if os.path.exists(sign_path):
                        # Load and resize image
                        img = cv2.imread(sign_path)
                        if img is not None:
                            img_resized = cv2.resize(img, self.config.VIDEO_RESOLUTION)
                            
                            # Add multiple frames for duration
                            num_frames = int(frame_duration * fps) or 1
                            for _ in range(num_frames):
                                frames.append(img_resized)
                        else:
                            logger.warning(f"Could not load image: {sign_path}")
                    else:
                        logger.warning(f"Sign image not found: {sign_path}")
                
                processed_chars += 1
                progress = 40 + (processed_chars / total_chars) * 40
                processing_status.update_progress(task_id, progress, f"Processing character {processed_chars}/{total_chars}")
            
            if not frames:
                raise ValueError("No valid frames generated")
            
            processing_status.update_progress(task_id, 85, "Creating video file")
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sign_video_{task_id}_{timestamp}.mp4"
            output_path = os.path.join(self.output_path, filename)
            
            # Create video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, self.config.VIDEO_RESOLUTION)
            
            for frame in frames:
                video_writer.write(frame)
            
            video_writer.release()
            
            processing_status.update_progress(task_id, 100, "Video generation completed")
            
            # Verify file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Video created successfully: {output_path}")
                return filename
            else:
                raise Exception("Video file was not created or is empty")
                
        except Exception as e:
            logger.error(f"Error in text_to_video: {str(e)}")
            raise e
    
    def _get_available_signs(self) -> List[str]:
        """Get list of available sign images"""
        signs = []
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            sign_path = os.path.join(self.sign_images_path, f"{letter}.png")
            if os.path.exists(sign_path):
                signs.append(letter)
        return signs

# Initialize converter
config = Config()
converter = SignLanguageConverter(config)

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Text-to-Sign Pipeline",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/convert-text-to-sign", "/task-status", "/download-video", "/generate-quiz"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Text-to-Sign Pipeline",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "available_signs": len(converter._get_available_signs())
    }

@app.post("/convert-text-to-sign", response_model=TextToSignResponse)
async def convert_text_to_sign(request: TextToSignRequest, background_tasks: BackgroundTasks):
    """Convert text to sign language video"""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        if len(request.text) > config.MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Text too long. Maximum length is {config.MAX_TEXT_LENGTH} characters"
            )
        
        # Generate unique task ID
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Initialize task status
        processing_status.start_new_task(task_id)
        
        # Start background processing
        background_tasks.add_task(
            process_text_to_sign,
            request.text,
            task_id,
            request.fps,
            request.frame_duration
        )
        
        return TextToSignResponse(
            task_id=task_id,
            status="processing",
            message="Text-to-sign conversion started",
            processing_details={
                "text_length": len(request.text),
                "estimated_duration": f"{len([c for c in request.text if c.isalpha()]) * request.frame_duration:.1f}s"
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting text-to-sign conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_text_to_sign(text: str, task_id: str, fps: float, frame_duration: float):
    """Background task to process text-to-sign conversion"""
    try:
        filename = converter.text_to_video(text, task_id, fps, frame_duration)
        processing_status.complete_task(task_id, {
            "video_filename": filename,
            "video_url": f"/download-video/{filename}",
            "text_processed": text[:100] + "..." if len(text) > 100 else text
        })
    except Exception as e:
        logger.error(f"Background processing error: {str(e)}")
        processing_status.complete_task(task_id, {"error": str(e)})

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get processing status for a task"""
    status = processing_status.get_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@app.get("/download-video/{filename}")
async def download_video(filename: str):
    """Download generated video file"""
    file_path = os.path.join(config.OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=file_path,
        media_type='video/mp4',
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/available-signs")
async def get_available_signs():
    """Get list of available sign language letters"""
    return {
        "available_signs": converter._get_available_signs(),
        "total_count": len(converter._get_available_signs())
    }

if __name__ == "__main__":
    logger.info("Starting Text-to-Sign Pipeline server...")
    
    # Cleanup old tasks on startup
    processing_status.cleanup_old_tasks()
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="info"
    )



