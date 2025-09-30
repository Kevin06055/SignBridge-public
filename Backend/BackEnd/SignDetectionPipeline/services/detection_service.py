import cv2
import numpy as np
import time
import asyncio
from typing import List, Tuple, Optional, Dict
import structlog
from pathlib import Path
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor

# YOLO and Supervision imports
from ultralytics import YOLO
import supervision as sv

from ..models import DetectedSign, BoundingBox, SignDetectionResponse, DetectionMethod
from shared.monitoring import monitor_requests
from shared.config import settings

logger = structlog.get_logger()

class YOLOSignDetectionService:
    def __init__(self, model_path: str = "models/SignConv.pt"):
        """Initialize YOLO model and supervision annotators"""
        self.model_path = model_path
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Supervision annotators for beautiful visualizations
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=0.8
        )
        self.label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_thickness=2,
            text_scale=0.8,
            text_color=sv.Color.WHITE
        )
        
        # Load model
        self._load_model()
        
        # ASL letter mapping (customize based on your model classes)
        self.class_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
    
    def _load_model(self):
        """Load the trained YOLO model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info("YOLO SignConv model loaded", model_path=self.model_path)
            else:
                logger.warning("SignConv.pt not found, using default YOLOv8 model")
                self.model = YOLO('yolov8n.pt')  # Fallback to default model
                
        except Exception as e:
            logger.error("Failed to load YOLO model", error=str(e))
            raise Exception(f"Model loading failed: {str(e)}")
    
    @monitor_requests("sign_detection")
    async def detect_signs_from_image(
        self, 
        image_bytes: bytes, 
        confidence_threshold: float = 0.8,
        max_detections: int = 10
    ) -> Tuple[SignDetectionResponse, bytes]:
        """
        Detect sign language from image using YOLO
        Returns both detection results and annotated image
        """
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._process_image_yolo,
                image_bytes,
                confidence_threshold,
                max_detections
            )
            
            detected_signs, annotated_image_bytes = result
            transcription = self._generate_transcription(detected_signs)
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            response = SignDetectionResponse(
                detected_signs=detected_signs,
                transcription=transcription,
                processing_time_ms=processing_time_ms,
                method_used=DetectionMethod.IMAGE,
                total_detections=len(detected_signs)
            )
            
            logger.info(
                "YOLO sign detection completed",
                detections_count=len(detected_signs),
                transcription=transcription,
                processing_time_ms=processing_time_ms
            )
            
            return response, annotated_image_bytes
            
        except Exception as e:
            logger.error("YOLO sign detection failed", error=str(e))
            raise Exception(f"Sign detection failed: {str(e)}")
    
    def _process_image_yolo(
        self, 
        image_bytes: bytes, 
        confidence_threshold: float,
        max_detections: int
    ) -> Tuple[List[DetectedSign], bytes]:
        """Process image using YOLO model with supervision annotations"""
        
        # Convert bytes to cv2 image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image format")
        
        # Run YOLO inference
        results = self.model(image, conf=confidence_threshold)[0]
        
        # Convert to supervision Detection format
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter by max detections
        if len(detections) > max_detections:
            # Sort by confidence and take top N
            confidence_indices = np.argsort(detections.confidence)[::-1][:max_detections]
            detections = detections[confidence_indices]
        
        # Create DetectedSign objects
        detected_signs = []
        labels = []
        
        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            confidence = detections.confidence[i]
            class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
            
            # Get letter from class name
            letter = self.class_names[class_id] if class_id < len(self.class_names) else 'A'
            
            # Create bounding box
            x1, y1, x2, y2 = xyxy.astype(int)
            width = x2 - x1
            height = y2 - y1
            
            detected_sign = DetectedSign(
                letter=letter,
                confidence=float(confidence),
                bounding_box=BoundingBox(x=int(x1), y=int(y1), width=int(width), height=int(height))
            )
            detected_signs.append(detected_sign)
            
            # Create label for annotation
            labels.append(f"{letter} {confidence:.2f}")
        
        # Create annotated image using supervision
        annotated_image = image.copy()
        if len(detections) > 0:
            # Add bounding boxes
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image,
                detections=detections
            )
            
            # Add labels
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image,
                detections=detections,
                labels=labels
            )
        
        # Convert annotated image back to bytes
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_bytes = buffer.tobytes()
        
        return detected_signs, annotated_image_bytes
    
    @monitor_requests("sign_detection")
    async def detect_signs_from_video(
        self, 
        video_bytes: bytes,
        confidence_threshold: float = 0.8,
        frame_interval: int = 5,
        max_duration_seconds: int = 30
    ) -> Tuple[List[SignDetectionResponse], bytes]:
        """
        Detect signs from video and return annotated video
        """
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._process_video_yolo,
                video_bytes,
                confidence_threshold,
                frame_interval,
                max_duration_seconds
            )
            
            frame_results, annotated_video_bytes = result
            
            logger.info(
                "YOLO video sign detection completed",
                frames_processed=len(frame_results),
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
            
            return frame_results, annotated_video_bytes
            
        except Exception as e:
            logger.error("YOLO video sign detection failed", error=str(e))
            raise Exception(f"Video sign detection failed: {str(e)}")
    
    def _process_video_yolo(
        self, 
        video_bytes: bytes,
        confidence_threshold: float,
        frame_interval: int,
        max_duration_seconds: int
    ) -> Tuple[List[SignDetectionResponse], bytes]:
        """Process video using YOLO with supervision annotations"""
        
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_bytes)
            video_path = temp_file.name
        
        # Output path for annotated video
        output_path = video_path.replace('.mp4', '_annotated.mp4')
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Limit processing duration
            max_frames = min(total_frames, int(fps * max_duration_seconds))
            
            # Video writer for annotated output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_results = []
            frame_count = 0
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame for detection
                if frame_count % frame_interval == 0:
                    # Run YOLO inference
                    results = self.model(frame, conf=confidence_threshold)[0]
                    detections = sv.Detections.from_ultralytics(results)
                    
                    # Create detected signs
                    detected_signs = []
                    labels = []
                    
                    for i in range(len(detections)):
                        xyxy = detections.xyxy[i]
                        confidence = detections.confidence[i]
                        class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
                        
                        letter = self.class_names[class_id] if class_id < len(self.class_names) else 'A'
                        
                        x1, y1, x2, y2 = xyxy.astype(int)
                        width = x2 - x1
                        height = y2 - y1
                        
                        detected_sign = DetectedSign(
                            letter=letter,
                            confidence=float(confidence),
                            bounding_box=BoundingBox(x=int(x1), y=int(y1), width=int(width), height=int(height))
                        )
                        detected_signs.append(detected_sign)
                        labels.append(f"{letter} {confidence:.2f}")
                    
                    if detected_signs:
                        transcription = self._generate_transcription(detected_signs)
                        frame_result = SignDetectionResponse(
                            detected_signs=detected_signs,
                            transcription=transcription,
                            processing_time_ms=0,
                            method_used=DetectionMethod.VIDEO,
                            total_detections=len(detected_signs)
                        )
                        frame_results.append(frame_result)
                    
                    # Annotate frame
                    if len(detections) > 0:
                        frame = self.box_annotator.annotate(scene=frame, detections=detections)
                        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                
                # Write frame to output video
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            # Read annotated video as bytes
            with open(output_path, 'rb') as f:
                annotated_video_bytes = f.read()
            
            return frame_results, annotated_video_bytes
            
        finally:
            # Cleanup temporary files
            for path in [video_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def _generate_transcription(self, detected_signs: List[DetectedSign]) -> str:
        """Generate text transcription from detected signs"""
        if not detected_signs:
            return ""
        
        # Sort by x-coordinate (left to right)
        sorted_signs = sorted(detected_signs, key=lambda s: s.bounding_box.x)
        letters = [sign.letter for sign in sorted_signs]
        return " ".join(letters)
