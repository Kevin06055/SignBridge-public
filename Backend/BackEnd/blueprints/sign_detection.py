"""
Sign Detection Blueprint
Handles real-time sign language detection using YOLO models
"""

from flask import Blueprint, request, jsonify, Response
import cv2
import numpy as np
import os
import threading
import base64
import time
import logging
from collections import deque
from typing import Optional, Dict, Any, List
import json

# Create blueprint
sign_detection_bp = Blueprint('sign_detection', __name__)

# Global variables for model and processing
model_global = None
prediction_history_global = deque(maxlen=5)
current_prediction = None
model_lock = threading.Lock()

def load_yolo_model():
    """Load YOLO model with fallback"""
    global model_global
    
    if model_global is not None:
        return model_global
    
    try:
        from ultralytics import YOLO
        from flask import current_app
        
        # Try to load the fine-tuned model first
        model_path = current_app.config.get('YOLO_MODEL_PATH')
        fallback_path = current_app.config.get('YOLO_FALLBACK_MODEL')
        
        if model_path and os.path.exists(model_path):
            logging.info(f"Loading fine-tuned YOLO model: {model_path}")
            model_global = YOLO(str(model_path))
        elif fallback_path and os.path.exists(fallback_path):
            logging.info(f"Loading fallback YOLO model: {fallback_path}")
            model_global = YOLO(str(fallback_path))
        else:
            logging.info("Loading default YOLOv8n model")
            model_global = YOLO('yolov8n.pt')
        
        return model_global
        
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        raise RuntimeError(f"Failed to load YOLO model: {e}")

def apply_preprocessing(image):
    """Apply optimized preprocessing pipeline"""
    try:
        # Step 1: Apply bilateral filter for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Step 2: Apply contrast enhancement using CLAHE
        lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 3: Apply skin segmentation
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        skin_segmented = cv2.bitwise_and(enhanced, enhanced, mask=skin_mask)
        
        return {
            'original': image,
            'bilateral': bilateral,
            'enhanced': enhanced,
            'skin_mask': skin_mask,
            'skin_segmented': skin_segmented,
            'preprocessed': skin_segmented 
        }
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        return {
            'original': image,
            'bilateral': image,
            'enhanced': image,
            'skin_mask': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),
            'skin_segmented': image,
            'preprocessed': image
        }

def create_visualization_grid(processed_results):
    """Create a 2x2 grid of preprocessing steps"""
    try:
        # Get image dimensions
        h, w = processed_results['original'].shape[:2]
        
        # Convert mask to BGR for visualization
        skin_mask_bgr = cv2.cvtColor(processed_results['skin_mask'], cv2.COLOR_GRAY2BGR)
        
        # Create the grid
        top_row = np.hstack([processed_results['original'], processed_results['bilateral']])
        bottom_row = np.hstack([processed_results['enhanced'], processed_results['skin_segmented']])
        
        grid = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(grid, "Original", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(grid, "Bilateral Filter", (w + 10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(grid, "Contrast Enhanced", (10, h + 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(grid, "Skin Segmented", (w + 10, h + 30), font, 0.7, (0, 255, 0), 2)
        
        return grid
    except Exception as e:
        logging.error(f"Error creating visualization grid: {e}")
        return processed_results['original']

def process_frame_api(frame_data, conf_threshold=0.5, apply_smoothing=True):
    """Process a frame from API request and return detection results"""
    global model_global, prediction_history_global, current_prediction
    
    try:
        with model_lock:
            # Ensure model is loaded
            if model_global is None:
                model_global = load_yolo_model()
        
        # Decode base64 image
        if isinstance(frame_data, str):
            # Remove data URL prefix if present
            if 'base64,' in frame_data:
                frame_data = frame_data.split('base64,')[1]
            
            # Decode base64
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            frame = frame_data
        
        if frame is None:
            raise ValueError("Could not decode frame data")
        
        # Apply preprocessing
        processed_results = apply_preprocessing(frame)
        preprocessed_frame = processed_results['preprocessed']
        
        # Run YOLO detection
        results = model_global(preprocessed_frame, conf=conf_threshold)
        
        # Process results
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names.get(class_id, f"class_{class_id}")
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'class_name': class_name
                    })
        
        # Apply temporal smoothing if enabled
        if apply_smoothing:
            prediction_history_global.append(detections)
            
            # Simple voting-based smoothing
            if len(prediction_history_global) >= 3:
                # Get most frequent prediction
                class_votes = {}
                for hist_detections in prediction_history_global:
                    if hist_detections:
                        best_detection = max(hist_detections, key=lambda x: x['confidence'])
                        class_name = best_detection['class_name']
                        class_votes[class_name] = class_votes.get(class_name, 0) + 1
                
                if class_votes:
                    most_frequent_class = max(class_votes.keys(), key=lambda x: class_votes[x])
                    current_prediction = most_frequent_class
                else:
                    current_prediction = None
            else:
                if detections:
                    best_detection = max(detections, key=lambda x: x['confidence'])
                    current_prediction = best_detection['class_name']
                else:
                    current_prediction = None
        else:
            if detections:
                best_detection = max(detections, key=lambda x: x['confidence'])
                current_prediction = best_detection['class_name']
            else:
                current_prediction = None
        
        # Encode processed frame for response
        _, buffer = cv2.imencode('.jpg', preprocessed_frame)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create visualization grid
        viz_grid = create_visualization_grid(processed_results)
        _, viz_buffer = cv2.imencode('.jpg', viz_grid)
        visualization_b64 = base64.b64encode(viz_buffer).decode('utf-8')
        
        return {
            'success': True,
            'detections': detections,
            'current_prediction': current_prediction,
            'confidence_threshold': conf_threshold,
            'frame_processed': True,
            'processed_frame': processed_frame_b64,
            'visualization': visualization_b64,
            'timestamp': time.time()
        }
        
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return {
            'success': False,
            'error': str(e),
            'detections': [],
            'current_prediction': None,
            'timestamp': time.time()
        }

# Blueprint routes
@sign_detection_bp.route('/detect', methods=['POST'])
def detect_sign():
    """Detect sign language from uploaded image or base64 data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get frame data
        frame_data = data.get('frame') or data.get('image')
        if not frame_data:
            return jsonify({'error': 'No frame or image data provided'}), 400
        
        # Get configuration
        from flask import current_app
        conf_threshold = data.get('confidence', current_app.config.get('YOLO_CONFIDENCE_THRESHOLD', 0.5))
        apply_smoothing = data.get('smoothing', True)
        
        # Process frame
        result = process_frame_api(frame_data, conf_threshold, apply_smoothing)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in detect_sign: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@sign_detection_bp.route('/detect-realtime', methods=['POST'])
def detect_sign_realtime():
    """Real-time sign detection for streaming"""
    try:
        data = request.get_json()
        
        if not data or 'frame' not in data:
            return jsonify({'error': 'Frame data is required'}), 400
        
        # Get configuration
        from flask import current_app
        conf_threshold = data.get('confidence', current_app.config.get('YOLO_CONFIDENCE_THRESHOLD', 0.5))
        
        # Process frame with smoothing enabled
        result = process_frame_api(data['frame'], conf_threshold, apply_smoothing=True)
        
        # Return minimal response for real-time processing
        return jsonify({
            'success': result['success'],
            'current_prediction': result.get('current_prediction'),
            'detections': result.get('detections', []),
            'confidence_threshold': conf_threshold,
            'timestamp': result.get('timestamp')
        })
        
    except Exception as e:
        logging.error(f"Error in detect_sign_realtime: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'current_prediction': None
        }), 500

@sign_detection_bp.route('/model/info')
def model_info():
    """Get information about the loaded model"""
    try:
        with model_lock:
            if model_global is None:
                load_yolo_model()
        
        from flask import current_app
        
        model_path = current_app.config.get('YOLO_MODEL_PATH')
        fallback_path = current_app.config.get('YOLO_FALLBACK_MODEL')
        
        return jsonify({
            'success': True,
            'model_info': {
                'model_loaded': model_global is not None,
                'model_path': str(model_path) if model_path else None,
                'fallback_path': str(fallback_path) if fallback_path else None,
                'model_exists': os.path.exists(model_path) if model_path else False,
                'fallback_exists': os.path.exists(fallback_path) if fallback_path else False,
                'class_names': getattr(model_global, 'names', {}) if model_global else {}
            }
        })
        
    except Exception as e:
        logging.error(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@sign_detection_bp.route('/reset-history', methods=['POST'])
def reset_prediction_history():
    """Reset prediction history for smoothing"""
    try:
        global prediction_history_global, current_prediction
        
        prediction_history_global.clear()
        current_prediction = None
        
        return jsonify({
            'success': True,
            'message': 'Prediction history reset'
        })
        
    except Exception as e:
        logging.error(f"Error resetting history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@sign_detection_bp.route('/health')
def health_check():
    """Health check for sign detection service"""
    try:
        from flask import current_app
        
        # Check if model can be loaded
        model_status = 'unknown'
        model_error = None
        
        try:
            with model_lock:
                if model_global is None:
                    load_yolo_model()
                model_status = 'loaded' if model_global is not None else 'failed'
        except Exception as e:
            model_status = 'failed'
            model_error = str(e)
        
        model_path = current_app.config.get('YOLO_MODEL_PATH')
        fallback_path = current_app.config.get('YOLO_FALLBACK_MODEL')
        
        return jsonify({
            'status': 'healthy' if model_status == 'loaded' else 'unhealthy',
            'service': 'sign-detection',
            'model_status': model_status,
            'model_error': model_error,
            'model_path': str(model_path) if model_path else None,
            'model_exists': os.path.exists(model_path) if model_path else False,
            'fallback_path': str(fallback_path) if fallback_path else None,
            'fallback_exists': os.path.exists(fallback_path) if fallback_path else False,
            'prediction_history_size': len(prediction_history_global),
            'current_prediction': current_prediction,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'sign-detection',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@sign_detection_bp.route('/configuration')
def get_configuration():
    """Get current configuration settings"""
    try:
        from flask import current_app
        
        return jsonify({
            'success': True,
            'configuration': {
                'confidence_threshold': current_app.config.get('YOLO_CONFIDENCE_THRESHOLD', 0.5),
                'model_path': str(current_app.config.get('YOLO_MODEL_PATH', '')),
                'fallback_model': str(current_app.config.get('YOLO_FALLBACK_MODEL', '')),
                'smoothing_enabled': True,
                'history_size': prediction_history_global.maxlen,
                'preprocessing_enabled': True
            }
        })
        
    except Exception as e:
        logging.error(f"Error getting configuration: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500