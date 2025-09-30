import cv2
import numpy as np
import argparse
import time
import os
from ultralytics import YOLO
from collections import deque

def parse_args():
    parser = argparse.ArgumentParser(description='Real-time sign language detection with optimized preprocessing')
    parser.add_argument('--model', type=str, default='fine_tuned.pt', help='Path to YOLO model file')
    parser.add_argument('--source', type=int, default=0, help='Camera device number')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--show-preprocessing', action='store_true', help='Show preprocessing steps')
    parser.add_argument('--record', type=str, default='', help='Path to save video recording')
    parser.add_argument('--smooth', type=int, default=5, help='Frames for temporal smoothing (0 to disable)')
    return parser.parse_args()

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
    
    return {
        'original': image,
        'bilateral': bilateral,
        'enhanced': enhanced,
        'skin_mask': skin_mask,
        'skin_segmented': skin_segmented
    }

def create_visualization_grid(processed_results):
    """Create a 2x2 grid of preprocessing steps"""
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

def main():
    args = parse_args()
    
    # Load YOLO model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Initialize camera
    print(f"Opening camera device {args.source}...")
    cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Failed to open camera device {args.source}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define ROI for hand detection
    roi_size = 300
    center_x, center_y = width // 2, height // 2
    roi_x1 = center_x - roi_size // 2
    roi_y1 = center_y - roi_size // 2
    roi_x2 = center_x + roi_size // 2
    roi_y2 = center_y + roi_size // 2
    
    # Initialize video writer if recording
    out = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.record, fourcc, fps, (width, height))
    
    # Load class names
    class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    if os.path.exists('data/classes.txt'):
        with open('data/classes.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    
    # Initialize variables for FPS calculation
    frame_count = 0
    fps_start_time = time.time()
    display_fps = 0
    
    # Prediction history for temporal smoothing
    prediction_history = deque(maxlen=args.smooth if args.smooth > 0 else 1)
    
    print("Running inference. Press 'q' to quit.")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame from camera")
            break
        
        # Mirror the frame (more intuitive for sign language)
        frame = cv2.flip(frame, 1)
        
        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        if roi.size == 0:
            continue
        
        # Apply preprocessing
        processed = apply_preprocessing(roi)
        
        # Run inference on the preprocessed ROI
        results = model(processed['skin_segmented'])[0]
        
        # Extract detections above confidence threshold
        detections = []
        for det in results.boxes:
            conf = det.conf.item()
            if conf > args.conf:
                cls_id = int(det.cls.item())
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                detections.append((cls_id, conf, (x1, y1, x2, y2)))
        
        # Apply temporal smoothing if enabled
        if args.smooth > 0 and detections:
            # Get highest confidence detection
            best_detection = max(detections, key=lambda x: x[1])
            prediction_history.append(best_detection[0])
            
            # Count class occurrences in history
            if len(prediction_history) >= args.smooth // 2:
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
                            # Replace all detections with just this one
                            detections = [det]
                            break
        
        # Draw ROI on the frame
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        
        # Draw detections on the ROI
        annotated_roi = roi.copy()
        
        if detections:
            # Get highest confidence detection
            cls_id, conf, (x1, y1, x2, y2) = max(detections, key=lambda x: x[1])
            class_label = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            
            # Draw bounding box
            cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw a large centered label
            text_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_DUPLEX, 2, 2)[0]
            text_x = (roi_size - text_size[0]) // 2
            text_y = (roi_size + text_size[1]) // 2
            
            # Draw background for text
            cv2.rectangle(annotated_roi, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(annotated_roi, class_label, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)
            
            # Add confidence
            conf_text = f"Conf: {conf:.2f}"
            cv2.putText(annotated_roi, conf_text, (text_x, text_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Insert annotated ROI back into the frame
        frame[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_roi
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 1.0:
            display_fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()
        
        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame to output video if recording
        if out is not None:
            out.write(frame)
        
        # Show main frame
        cv2.imshow("Sign Language Detection", frame)
        
        # Show preprocessing steps if enabled
        if args.show_preprocessing:
            grid = create_visualization_grid(processed)
            cv2.imshow("Preprocessing Steps", grid)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out is not None:
        out.write(frame)
    cv2.destroyAllWindows()
    
    print("Inference stopped.")

if __name__ == "__main__":
    main()