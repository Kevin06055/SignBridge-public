import cv2
import numpy as np
import os
import time
import argparse
from ultralytics import YOLO
from collections import Counter, deque

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze sign language detection with various preprocessing techniques')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to YOLO model file')
    parser.add_argument('--input', type=str, default='0', help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--output', type=str, default='', help='Output video file path (optional)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--roi-size', type=int, default=300, help='Size of ROI square')
    parser.add_argument('--show-grid', action='store_true', help='Show preprocessing visualization grid')
    parser.add_argument('--save-results', action='store_true', help='Save preprocessing analysis images')
    parser.add_argument('--result-dir', type=str, default='preprocessing_results', help='Directory to save result images')
    return parser.parse_args()

class PreprocessingAnalyzer:
    def __init__(self, model_path, conf_threshold=0.5, temporal_window=5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.temporal_window = temporal_window
        self.prediction_history = deque(maxlen=temporal_window)
        
        # Default class names
        self.class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Load class names from file if available
        if os.path.exists('data/classes.txt'):
            with open('data/classes.txt', 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
    
    def apply_preprocessing_suite(self, image):
        """Apply multiple preprocessing techniques to the input image"""
        results = {}
        
        # Store original image
        results['original'] = image.copy()
        
        # 1. Basic enhancements
        results['gray'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results['gray_bgr'] = cv2.cvtColor(results['gray'], cv2.COLOR_GRAY2BGR)
        
        # 2. Skin segmentation
        results['skin_mask'], results['skin_segmented'] = self.apply_skin_segmentation(image)
        
        # 3. Contrast enhancement
        results['contrast_enhanced'] = self.apply_contrast_enhancement(image)
        
        # 4. Edge detection
        results['edges'] = self.apply_edge_detection(image)
        
        # 5. Bilateral filtering (noise reduction while preserving edges)
        results['bilateral'] = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 6. Adaptive thresholding
        gray = results['gray']
        results['adaptive_thresh'] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        results['adaptive_thresh_bgr'] = cv2.cvtColor(results['adaptive_thresh'], cv2.COLOR_GRAY2BGR)
        
        # 7. Color filtering
        results['color_filtered'] = self.apply_color_filtering(image)
        
        # 8. Combined preprocessing (bilateral + contrast + skin segmentation)
        bilateral = results['bilateral']
        contrast_enhanced = self.apply_contrast_enhancement(bilateral)
        _, skin_segmented = self.apply_skin_segmentation(contrast_enhanced)
        results['combined'] = skin_segmented
        
        return results
    
    def apply_skin_segmentation(self, image):
        """Apply skin segmentation to focus on hand regions"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Add additional skin tone range
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask, mask2)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply the mask to the original image
        skin_segmented = cv2.bitwise_and(image, image, mask=mask)
        
        return mask, skin_segmented
    
    def apply_contrast_enhancement(self, image):
        """Enhance contrast using CLAHE"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split the LAB image into different channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L channel with the original A and B channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR color space
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def apply_edge_detection(self, image):
        """Apply edge detection to highlight hand contours"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert back to BGR for consistency
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edges_bgr
    
    def apply_color_filtering(self, image):
        """Filter specific color ranges common in sign language datasets"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Filter for skin and bright colored markers if any
        lower_bound = np.array([0, 30, 60])
        upper_bound = np.array([30, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        color_filtered = cv2.bitwise_and(image, image, mask=mask)
        
        return color_filtered
    
    def run_inference_on_all(self, preprocessed_images):
        """Run model inference on all preprocessed versions"""
        results = {}
        
        for name, img in preprocessed_images.items():
            # Skip grayscale images that haven't been converted to BGR
            if len(img.shape) < 3 or img.shape[2] != 3:
                continue
                
            # Run inference
            detections = self.model(img)[0]
            
            # Extract detections above confidence threshold
            boxes = []
            for det in detections.boxes:
                conf = det.conf.item()
                if conf > self.conf_threshold:
                    cls_id = int(det.cls.item())
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    boxes.append((cls_id, conf, (x1, y1, x2, y2)))
            
            results[name] = boxes
        
        return results
    
    def create_visualization_grid(self, preprocessed_images):
        """Create a visualization grid of all preprocessing techniques"""
        # Get dimensions of the first image
        first_img = list(preprocessed_images.values())[0]
        h, w = first_img.shape[:2]
        
        # Convert grayscale images to BGR for display
        for name, img in preprocessed_images.items():
            if len(img.shape) == 2:
                preprocessed_images[name] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Create a 3x3 grid
        top_row = np.hstack([
            preprocessed_images['original'], 
            preprocessed_images['contrast_enhanced'], 
            preprocessed_images['bilateral']
        ])
        
        middle_row = np.hstack([
            preprocessed_images['skin_segmented'],
            preprocessed_images['edges'],
            preprocessed_images['adaptive_thresh_bgr']
        ])
        
        bottom_row = np.hstack([
            preprocessed_images['color_filtered'],
            preprocessed_images['combined'],
            preprocessed_images['gray_bgr']
        ])
        
        grid = np.vstack([top_row, middle_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        labels = [
            ('Original', 10, 30), 
            ('Contrast Enhanced', w + 10, 30),
            ('Bilateral Filter', 2*w + 10, 30),
            ('Skin Segmented', 10, h + 30), 
            ('Edge Detection', w + 10, h + 30),
            ('Adaptive Threshold', 2*w + 10, h + 30),
            ('Color Filtered', 10, 2*h + 30), 
            ('Combined', w + 10, 2*h + 30),
            ('Grayscale', 2*w + 10, 2*h + 30)
        ]
        
        for text, x, y in labels:
            cv2.putText(grid, text, (x, y), font, 0.7, (0, 255, 0), 2)
        
        return grid
    
    def create_detection_grid(self, preprocessed_images, inference_results):
        """Create a grid showing detections on each preprocessed image"""
        # Create copies of images to draw on
        result_images = {}
        
        for name, img in preprocessed_images.items():
            # Skip grayscale images
            if len(img.shape) < 3 or img.shape[2] != 3:
                continue
                
            # Create a copy
            result_img = img.copy()
            
            # Draw detections
            for cls_id, conf, (x1, y1, x2, y2) in inference_results.get(name, []):
                # Get class name
                class_label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
                
                # Draw bounding box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_label} {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(result_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
                cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            result_images[name] = result_img
        
        # Create a 3x3 grid similar to the preprocessing grid
        h, w = result_images['original'].shape[:2]
        
        top_row = np.hstack([
            result_images.get('original', np.zeros((h, w, 3), dtype=np.uint8)), 
            result_images.get('contrast_enhanced', np.zeros((h, w, 3), dtype=np.uint8)), 
            result_images.get('bilateral', np.zeros((h, w, 3), dtype=np.uint8))
        ])
        
        middle_row = np.hstack([
            result_images.get('skin_segmented', np.zeros((h, w, 3), dtype=np.uint8)),
            result_images.get('edges', np.zeros((h, w, 3), dtype=np.uint8)),
            result_images.get('adaptive_thresh_bgr', np.zeros((h, w, 3), dtype=np.uint8))
        ])
        
        bottom_row = np.hstack([
            result_images.get('color_filtered', np.zeros((h, w, 3), dtype=np.uint8)),
            result_images.get('combined', np.zeros((h, w, 3), dtype=np.uint8)),
            np.zeros((h, w, 3), dtype=np.uint8)  # Empty space for the grayscale
        ])
        
        grid = np.vstack([top_row, middle_row, bottom_row])
        
        # Add labels and detection counts
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        labels = [
            ('Original', 10, 30, len(inference_results.get('original', []))), 
            ('Contrast Enhanced', w + 10, 30, len(inference_results.get('contrast_enhanced', []))),
            ('Bilateral Filter', 2*w + 10, 30, len(inference_results.get('bilateral', []))),
            ('Skin Segmented', 10, h + 30, len(inference_results.get('skin_segmented', []))), 
            ('Edge Detection', w + 10, h + 30, len(inference_results.get('edges', []))),
            ('Adaptive Threshold', 2*w + 10, h + 30, len(inference_results.get('adaptive_thresh_bgr', []))),
            ('Color Filtered', 10, 2*h + 30, len(inference_results.get('color_filtered', []))), 
            ('Combined', w + 10, 2*h + 30, len(inference_results.get('combined', []))),
        ]
        
        for text, x, y, count in labels:
            cv2.putText(grid, f"{text} ({count})", (x, y), font, 0.7, (0, 255, 0), 2)
        
        return grid
    
    def find_best_prediction(self, inference_results):
        """Find the most confident prediction across all preprocessing methods"""
        all_predictions = []
        
        for method, detections in inference_results.items():
            for cls_id, conf, bbox in detections:
                all_predictions.append((method, cls_id, conf, bbox))
        
        if not all_predictions:
            return None, None, None, None
        
        # Sort by confidence (highest first)
        all_predictions.sort(key=lambda x: x[2], reverse=True)
        
        # Get the most confident prediction
        best_method, best_cls_id, best_conf, best_bbox = all_predictions[0]
        
        # Add to history for temporal smoothing
        self.prediction_history.append(best_cls_id)
        
        # Apply temporal smoothing
        if len(self.prediction_history) >= self.temporal_window // 2:
            counter = Counter(self.prediction_history)
            smoothed_cls_id, count = counter.most_common(1)[0]
            
            # If the smoothed prediction occurs in at least 60% of frames
            if count / len(self.prediction_history) >= 0.6:
                # Find the highest confidence for this class
                for method, cls_id, conf, bbox in all_predictions:
                    if cls_id == smoothed_cls_id:
                        return method, smoothed_cls_id, conf, bbox
        
        return best_method, best_cls_id, best_conf, best_bbox

def main():
    args = parse_args()
    
    # Initialize the preprocessing analyzer
    analyzer = PreprocessingAnalyzer(args.model, args.conf)
    
    # Create output directory if saving results
    if args.save_results:
        os.makedirs(args.result_dir, exist_ok=True)
    
    # Initialize video source
    try:
        if args.input.isdigit():
            source = int(args.input)
        else:
            source = args.input
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source {args.input}")
            return
        
        # Set camera properties for better capture if using webcam
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception as e:
        print(f"Error initializing video source: {e}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output is specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Define ROI coordinates
    center_x, center_y = width // 2, height // 2
    roi_size = args.roi_size
    roi_x1 = center_x - roi_size // 2
    roi_y1 = center_y - roi_size // 2
    roi_x2 = center_x + roi_size // 2
    roi_y2 = center_y + roi_size // 2
    
    # Variables for FPS calculation
    frame_count = 0
    fps_start_time = time.time()
    display_fps = 0
    
    print("Starting analysis. Press 'q' to quit, 's' to save current frame.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        if roi.size == 0:
            continue
        
        # Apply preprocessing suite
        preprocessed = analyzer.apply_preprocessing_suite(roi)
        
        # Run inference on all preprocessed images
        inference_results = analyzer.run_inference_on_all(preprocessed)
        
        # Find best prediction
        best_method, best_cls_id, best_conf, best_bbox = analyzer.find_best_prediction(inference_results)
        
        # Create visualization grids
        if args.show_grid:
            preprocessing_grid = analyzer.create_visualization_grid(preprocessed)
            detection_grid = analyzer.create_detection_grid(preprocessed, inference_results)
            
            # Resize grids if needed
            grid_height, grid_width = preprocessing_grid.shape[:2]
            if grid_width > 1280:
                scale = 1280 / grid_width
                preprocessing_grid = cv2.resize(preprocessing_grid, (0, 0), fx=scale, fy=scale)
                detection_grid = cv2.resize(detection_grid, (0, 0), fx=scale, fy=scale)
        
        # Draw ROI on the frame
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        
        # Display best prediction on frame
        if best_cls_id is not None:
            # Get class name
            class_label = analyzer.class_names[best_cls_id] if best_cls_id < len(analyzer.class_names) else f"Class {best_cls_id}"
            
            # Create overlay for the result
            overlay = frame.copy()
            
            # Draw large text for the sign
            text_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_DUPLEX, 2, 2)[0]
            text_x = 50
            text_y = height - 50
            
            # Draw background rectangle
            cv2.rectangle(overlay, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            # Draw prediction info
            cv2.putText(overlay, class_label, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)
            
            # Add confidence and method info
            info_text = f"Conf: {best_conf:.2f} | Method: {best_method}"
            cv2.putText(overlay, info_text, 
                       (text_x, text_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Blend overlay with frame
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 1.0:
            display_fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()
        
        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame to output video if specified
        if out is not None:
            out.write(frame)
        
        # Display results
        cv2.imshow("Sign Language Analysis", frame)
        
        if args.show_grid:
            cv2.imshow("Preprocessing Techniques", preprocessing_grid)
            cv2.imshow("Detection Results", detection_grid)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Quit on 'q'
        if key == ord('q'):
            break
        
        # Save current frame and analysis on 's'
        elif key == ord('s') and args.save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save main frame
            cv2.imwrite(os.path.join(args.result_dir, f"frame_{timestamp}.jpg"), frame)
            
            # Save preprocessing grid
            if args.show_grid:
                cv2.imwrite(os.path.join(args.result_dir, f"preprocessing_{timestamp}.jpg"), preprocessing_grid)
                cv2.imwrite(os.path.join(args.result_dir, f"detections_{timestamp}.jpg"), detection_grid)
            
            print(f"Saved analysis to {args.result_dir}")
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print("Analysis completed.")

if __name__ == "__main__":
    main()