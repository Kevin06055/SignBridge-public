import cv2
import numpy as np
import argparse
import time
import os
import threading
import base64
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
from collections import deque
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_global = None
prediction_history_global = deque(maxlen=5)
current_prediction = None
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

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
            'skin_segmented': skin_segmented
        }
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return {'skin_segmented': image, 'original': image}

def process_frame_api(frame_data, conf_threshold=0.6):
    """Process a frame from API request and return detection results"""
    global model_global, prediction_history_global, current_prediction
    
    try:
        # Handle different base64 formats
        if ',' in frame_data:
            encoded_data = frame_data.split(',')[1]
        else:
            encoded_data = frame_data
            
        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "error": "Failed to decode image"}
        
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Apply preprocessing
        processed = apply_preprocessing(frame)
        
        # Run inference
        results = model_global(processed['skin_segmented'])[0]
        
        # Extract detections above confidence threshold
        detections = []
        if results.boxes is not None:
            for det in results.boxes:
                conf = det.conf.item()
                if conf > conf_threshold:
                    cls_id = int(det.cls.item())
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    detections.append((cls_id, conf, (x1, y1, x2, y2)))
        
        # Apply temporal smoothing
        if detections:
            # Get highest confidence detection
            best_detection = max(detections, key=lambda x: x[1])
            prediction_history_global.append(best_detection[0])
            
            # Count class occurrences in history
            if len(prediction_history_global) >= 3:
                class_counts = {}
                for cls_id in prediction_history_global:
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                
                # Find most frequent class
                smoothed_cls_id = max(class_counts.items(), key=lambda x: x[1])[0]
                max_count = class_counts[smoothed_cls_id]
                
                # Use smoothed prediction if it appears in more than 60% of frames
                if max_count / len(prediction_history_global) >= 0.6:
                    # Find the detection with this class
                    for det in detections:
                        if det[0] == smoothed_cls_id:
                            class_label = class_names[det[0]] if det[0] < len(class_names) else f"Class {det[0]}"
                            current_prediction = {
                                "success": True,
                                "data": {
                                    "label": class_label,
                                    "confidence": float(det[1]),
                                    "bbox": list(det[2]),
                                    "timestamp": time.time()
                                }
                            }
                            return current_prediction
        
        # If no stable prediction, use the best detection
        if detections:
            best_det = max(detections, key=lambda x: x[1])
            class_label = class_names[best_det[0]] if best_det[0] < len(class_names) else f"Class {best_det[0]}"
            
            current_prediction = {
                "success": True,
                "data": {
                    "label": class_label,
                    "confidence": float(best_det[1]),
                    "bbox": list(best_det[2]),
                    "timestamp": time.time()
                }
            }
            return current_prediction
        
        # Return empty result if no detections
        return {
            "success": True,
            "data": {
                "label": "",
                "confidence": 0,
                "bbox": [],
                "timestamp": time.time()
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"success": False, "error": str(e)}

# API Routes matching frontend expectations
@app.route('/api/v1/sign-detection/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_global
    try:
        model_status = model_global is not None
        return jsonify({
            "success": True,
            "data": {
                "status": "healthy" if model_status else "unhealthy",
                "model_loaded": model_status,
                "num_classes": len(class_names),
                "timestamp": time.time()
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/sign-detection', methods=['POST'])
def detect_sign():
    """Main detection endpoint"""
    try:
        # Check if model is loaded
        if model_global is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded"
            }), 503
        
        # Get frame data from request
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({
                "success": False,
                "error": "No frame data provided"
            }), 400
        
        frame_data = data['frame']
        conf_threshold = float(data.get('confidence', 0.6))
        
        # Process frame
        result = process_frame_api(frame_data, conf_threshold)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Detection endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/sign-detection/status', methods=['GET'])
def get_status():
    """Status endpoint for debugging"""
    global model_global, prediction_history_global
    return jsonify({
        "success": True,
        "data": {
            "model_loaded": model_global is not None,
            "prediction_history_length": len(prediction_history_global),
            "available_classes": class_names,
            "server_time": time.time()
        }
    })

def initialize_model(model_path='best.pt'):
    """Initialize the YOLO model"""
    global model_global, class_names
    
    try:
        logger.info(f"Loading model from {model_path}...")
        model_global = YOLO(model_path)
        
        # Load class names from file if available
        if os.path.exists('data/classes.txt'):
            with open('data/classes.txt', 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
                
        logger.info(f"Model loaded successfully with {len(class_names)} classes")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

if __name__ == "__main__":
    # Initialize model
    model_path = os.getenv('MODEL_PATH', 'fine_tuned.pt')
    if initialize_model(model_path):
        # Start Flask server
        port = int(os.getenv('PORT', 5000))
        logger.info(f"Starting server on port {port}...")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        logger.error("Failed to initialize model. Server not started.")
