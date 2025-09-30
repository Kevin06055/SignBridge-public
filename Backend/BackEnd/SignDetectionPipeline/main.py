from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import structlog
import io
from contextlib import asynccontextmanager

from .models import SignDetectionResponse, VideoDetectionResponse
from .services.detection_service import YOLOSignDetectionService
from shared.config import settings
from shared.auth import verify_api_key
from shared.api_middleware import (
    APIResponseMiddleware, LoggingMiddleware, RateLimitMiddleware,
    create_success_response, create_error_response
)
from shared.api_models import ErrorCode, HealthCheckResponse

logger = structlog.get_logger()

# Global service
detection_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global detection_service
    
    logger.info("Starting YOLO Sign Detection Pipeline", version=settings.version)
    detection_service = YOLOSignDetectionService(model_path=settings.sign_detection_model_path)
    logger.info("YOLO Sign Detection service initialized")
    
    yield
    
    logger.info("Shutting down YOLO Sign Detection Pipeline")

# Create FastAPI app
app = FastAPI(
    title="YOLO Sign Detection Pipeline",
    description="Detect and transcribe sign language using YOLO and Supervision",
    version=settings.version,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    APIResponseMiddleware,
    service_name="SignDetection",
    service_version=settings.version
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    calls=50,  # 50 requests
    period=60  # per minute
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "yolo-sign-detection",
        "version": settings.version,
        "model_loaded": detection_service.model is not None
    }

@app.post("/detect/image", response_model=SignDetectionResponse)
async def detect_signs_from_image(
    file: UploadFile = File(..., description="Image file containing sign language"),
    confidence_threshold: float = Form(default=0.8, ge=0.0, le=1.0),
    max_detections: int = Form(default=10, ge=1, le=50),
    api_key: str = Depends(verify_api_key)
):
    """Detect sign language from uploaded image using YOLO"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        
        result, annotated_image_bytes = await detection_service.detect_signs_from_image(
            image_bytes=image_bytes,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections
        )
        
        return result
        
    except Exception as e:
        logger.error("YOLO image sign detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Sign detection failed")

@app.post("/detect/image/annotated")
async def detect_signs_with_annotations(
    file: UploadFile = File(..., description="Image file containing sign language"),
    confidence_threshold: float = Form(default=0.8, ge=0.0, le=1.0),
    max_detections: int = Form(default=10, ge=1, le=50),
    api_key: str = Depends(verify_api_key)
):
    """Detect signs and return annotated image with beautiful supervision annotations"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        
        result, annotated_image_bytes = await detection_service.detect_signs_from_image(
            image_bytes=image_bytes,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections
        )
        
        return StreamingResponse(
            io.BytesIO(annotated_image_bytes),
            media_type="image/jpeg",
            headers={"X-Detection-Count": str(result.total_detections)}
        )
        
    except Exception as e:
        logger.error("YOLO annotated image detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Sign detection failed")

@app.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_signs_from_video(
    file: UploadFile = File(..., description="Video file containing sign language"),
    confidence_threshold: float = Form(default=0.8, ge=0.0, le=1.0),
    frame_interval: int = Form(default=5, ge=1, le=30),
    max_duration_seconds: int = Form(default=30, ge=1, le=300),
    api_key: str = Depends(verify_api_key)
):
    """Detect sign language from uploaded video using YOLO"""
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        video_bytes = await file.read()
        
        frame_results, annotated_video_bytes = await detection_service.detect_signs_from_video(
            video_bytes=video_bytes,
            confidence_threshold=confidence_threshold,
            frame_interval=frame_interval,
            max_duration_seconds=max_duration_seconds
        )
        
        # Combine results
        all_transcriptions = [result.transcription for result in frame_results if result.transcription]
        combined_transcription = " ".join(all_transcriptions)
        
        response = VideoDetectionResponse(
            frame_results=frame_results,
            combined_transcription=combined_transcription,
            total_frames_processed=len(frame_results),
            video_duration_seconds=max_duration_seconds
        )
        
        return response
        
    except Exception as e:
        logger.error("YOLO video sign detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Video sign detection failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "SignDetectionPipeline.main:app",
        host=settings.sign_detection_host,
        port=settings.sign_detection_port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
