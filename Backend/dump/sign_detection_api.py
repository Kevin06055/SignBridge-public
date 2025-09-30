import os
import cv2
import numpy as np
import time
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
from collections import deque
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None
prediction_history = deque(maxlen=5)  # Store last 5 predictions for temporal smoothing
current_prediction = None
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Load class names from file if available
if os.path.exists('data/classes.txt'):
    with open('data/classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

def load_model(model_path='best.pt'):
    """Load the YOLO model"""
    global model
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def apply_preprocessing(image):
    """
    Apply optimized preprocessing pipeline
    """
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
    
    return skin_segmented

def process_frame(frame_data, conf_threshold=0.5):
    """Process a frame and return detection results"""
    global model, prediction_history, current_prediction
    
    # If model not loaded, load it
    if model is None:
        if not load_model():
            return {"error": "Failed to load model"}
    
    try:
        # Decode base64 image
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode image"}
        
        # Apply preprocessing
        processed_frame = apply_preprocessing(frame)
        
        # Run inference
        results = model(processed_frame)[0]
        
        # Extract detections above confidence threshold
        detections = []
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
            prediction_history.append(best_detection[0])
            
            # Count class occurrences in history
            class_counts = {}
            for cls_id in prediction_history:
                if cls_id not in class_counts:
                    class_counts[cls_id] = 0
                class_counts[cls_id] += 1
            
            # Find most frequent class
            max_count = 0
            smoothed_cls_id = None
            
            for cls_id, count in class_counts.items():
                if count > max_count:
                    max_count = count
                    smoothed_cls_id = cls_id
            
            # Use smoothed prediction if it appears in more than 60% of frames
            if smoothed_cls_id is not None and max_count / len(prediction_history) >= 0.6:
                # Find the detection with this class
                for det in detections:
                    if det[0] == smoothed_cls_id:
                        # Get class label
                        class_label = class_names[det[0]] if det[0] < len(class_names) else f"Class {det[0]}"
                        
                        # Update current prediction
                        current_prediction = {
                            "label": class_label,
                            "confidence": float(det[1]),
                            "bbox": list(det[2])
                        }
                        
                        # Return detection result
                        return current_prediction
            
            # If no stable prediction, use the best detection
            if detections:
                best_det = max(detections, key=lambda x: x[1])
                class_label = class_names[best_det[0]] if best_det[0] < len(class_names) else f"Class {best_det[0]}"
                
                current_prediction = {
                    "label": class_label,
                    "confidence": float(best_det[1]),
                    "bbox": list(best_det[2])
                }
                
                return current_prediction
        
        # Return the last prediction if no new detections
        return current_prediction if current_prediction else {"label": "", "confidence": 0, "bbox": []}
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {"error": str(e)}

@app.route('/api/detect', methods=['POST'])
def detect():
    """API endpoint for sign language detection"""
    if 'frame' not in request.json:
        return jsonify({"error": "No frame data provided"}), 400
    
    frame_data = request.json['frame']
    conf_threshold = float(request.json.get('conf_threshold', 0.5))
    
    result = process_frame(frame_data, conf_threshold)
    return jsonify(result)

@app.route('/api/status', methods=['GET'])
def status():
    """Check if the API is running and model is loaded"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        "status": "running",
        "model": model_status,
        "num_classes": len(class_names)
    })

if __name__ == '__main__':
    # Preload model
    load_model()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)