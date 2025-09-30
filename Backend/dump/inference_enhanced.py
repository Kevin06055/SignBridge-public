from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

# ----- CONFIGURATION -----
MODEL_PATH = "best.pt"  # path to your trained model
CONF_THRESHOLD = 0.5  # confidence threshold for displaying detections
ROI_SIZE = 300  # size of the square ROI for hand placement
LABELS_FILE = "data/classes.txt"  # path to labels file

# Load class names if available
class_names = []
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    # Default class names from data.yaml
    class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# ----- LOAD YOLO MODEL -----
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")

# ----- SETUP WEBCAM -----
print("Initializing webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Set camera properties for better capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure

# Read one frame to get dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read from webcam")
    exit()
height, width = frame.shape[:2]
center_x, center_y = width // 2, height // 2

# Define ROI coordinates
roi_x1 = center_x - ROI_SIZE // 2
roi_y1 = center_y - ROI_SIZE // 2
roi_x2 = center_x + ROI_SIZE // 2
roi_y2 = center_y + ROI_SIZE // 2

print("✅ Place your hand inside the green box. Press 'q' to quit.")

# ----- PREPROCESSING FUNCTIONS -----
def apply_skin_mask(image):
    """
    Creates a skin mask to focus on hand regions
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color (adjust these values based on lighting conditions)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Add additional skin tone range (handles some darker skin tones)
    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    
    # Combine masks
    mask = cv2.bitwise_or(mask, mask2)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 2)
    
    return mask

def enhance_contrast(image):
    """
    Enhances the contrast of the image to make hand features more visible
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into different channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the original A and B channels
    merged = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return enhanced

def preprocess_for_model(roi):
    """
    Apply a series of preprocessing steps to optimize the ROI for model inference
    """
    # Enhance contrast
    enhanced = enhance_contrast(roi)
    
    # Apply skin mask to focus on hand regions
    skin_mask = apply_skin_mask(enhanced)
    
    # Apply the mask to the original ROI
    masked_roi = cv2.bitwise_and(roi, roi, mask=skin_mask)
    
    # Convert to grayscale
    gray = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    # This helps in varying lighting conditions
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Edge detection to highlight hand contours
    edges = cv2.Canny(gray, 50, 150)
    
    # Combine grayscale image with edges for better feature detection
    edge_enhanced = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
    
    # Return multiple processed versions of the ROI
    return enhanced, masked_roi, cv2.cvtColor(edge_enhanced, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# ----- MAIN LOOP -----
frame_count = 0
fps_time = time.time()
fps = 0
last_predictions = []
prediction_buffer = []  # Buffer to stabilize predictions

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Calculate FPS
    frame_count += 1
    if (time.time() - fps_time) > 1:
        fps = frame_count / (time.time() - fps_time)
        frame_count = 0
        fps_time = time.time()
    
    # Mirror the frame for more intuitive interaction
    frame = cv2.flip(frame, 1)
    
    # Draw the ROI box on the frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    
    # Extract the ROI area
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        continue
    
    # Apply preprocessing to get enhanced versions of the ROI
    enhanced, masked_roi, edge_enhanced, thresholded = preprocess_for_model(roi)
    
    # Run YOLO inference on both the regular ROI and the enhanced versions
    results_original = model(roi)[0]
    results_enhanced = model(enhanced)[0]
    results_masked = model(masked_roi)[0]
    results_edge = model(edge_enhanced)[0]
    results_thresh = model(thresholded)[0]
    
    # Combine and process results from all versions
    all_boxes = []
    all_confidences = []
    all_class_ids = []
    
    # Process original ROI results
    for det in results_original.boxes:
        conf = det.conf.item()
        if conf > CONF_THRESHOLD:
            cls_id = int(det.cls.item())
            all_boxes.append(det.xyxy[0].tolist())
            all_confidences.append(conf)
            all_class_ids.append(cls_id)
    
    # Process enhanced results
    for det in results_enhanced.boxes:
        conf = det.conf.item()
        if conf > CONF_THRESHOLD:
            cls_id = int(det.cls.item())
            all_boxes.append(det.xyxy[0].tolist())
            all_confidences.append(conf)
            all_class_ids.append(cls_id)
    
    # Process masked results
    for det in results_masked.boxes:
        conf = det.conf.item()
        if conf > CONF_THRESHOLD:
            cls_id = int(det.cls.item())
            all_boxes.append(det.xyxy[0].tolist())
            all_confidences.append(conf)
            all_class_ids.append(cls_id)
    
    # Process edge enhanced results
    for det in results_edge.boxes:
        conf = det.conf.item()
        if conf > CONF_THRESHOLD:
            cls_id = int(det.cls.item())
            all_boxes.append(det.xyxy[0].tolist())
            all_confidences.append(conf)
            all_class_ids.append(cls_id)
    
    # Process thresholded results
    for det in results_thresh.boxes:
        conf = det.conf.item()
        if conf > CONF_THRESHOLD:
            cls_id = int(det.cls.item())
            all_boxes.append(det.xyxy[0].tolist())
            all_confidences.append(conf)
            all_class_ids.append(cls_id)
    
    # Count detections by class
    if all_class_ids:
        class_counts = {}
        for cls_id, conf in zip(all_class_ids, all_confidences):
            if cls_id not in class_counts:
                class_counts[cls_id] = []
            class_counts[cls_id].append(conf)
        
        # Find the most frequent class with the highest average confidence
        max_count = 0
        max_cls_id = -1
        max_avg_conf = 0
        
        for cls_id, confs in class_counts.items():
            count = len(confs)
            avg_conf = sum(confs) / count
            
            if count > max_count or (count == max_count and avg_conf > max_avg_conf):
                max_count = count
                max_cls_id = cls_id
                max_avg_conf = avg_conf
        
        # Add to prediction buffer for temporal smoothing
        if max_cls_id != -1:
            prediction_buffer.append((max_cls_id, max_avg_conf))
            if len(prediction_buffer) > 5:  # Keep last 5 predictions
                prediction_buffer.pop(0)
        
        # Temporal smoothing
        if prediction_buffer:
            buffer_counts = {}
            for cls_id, conf in prediction_buffer:
                if cls_id not in buffer_counts:
                    buffer_counts[cls_id] = []
                buffer_counts[cls_id].append(conf)
            
            # Find most frequent class in buffer
            max_buffer_count = 0
            final_cls_id = -1
            max_buffer_avg_conf = 0
            
            for cls_id, confs in buffer_counts.items():
                count = len(confs)
                avg_conf = sum(confs) / count
                
                if count > max_buffer_count or (count == max_buffer_count and avg_conf > max_buffer_avg_conf):
                    max_buffer_count = count
                    final_cls_id = cls_id
                    max_buffer_avg_conf = avg_conf
            
            # Final prediction
            if final_cls_id != -1 and max_buffer_avg_conf > 0.6:  # Higher threshold for smoothed prediction
                class_label = class_names[final_cls_id] if final_cls_id < len(class_names) else f"Class {final_cls_id}"
                prediction_text = f"{class_label} ({max_buffer_avg_conf:.2f})"
                last_predictions = [(final_cls_id, max_buffer_avg_conf)]
        
    # Create a copy of the ROI for annotation
    annotated_roi = roi.copy()
    
    # Highlight the highest confidence detection in the ROI
    if last_predictions:
        cls_id, conf = last_predictions[0]
        class_label = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        
        # Draw a large text in the center of the ROI
        text_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_DUPLEX, 2, 2)[0]
        text_x = (ROI_SIZE - text_size[0]) // 2
        text_y = (ROI_SIZE + text_size[1]) // 2
        
        # Draw a background rectangle for the text
        cv2.rectangle(annotated_roi, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(annotated_roi, class_label, (text_x, text_y),
                   cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)
        
        # Display confidence
        conf_text = f"Conf: {conf:.2f}"
        cv2.putText(annotated_roi, conf_text, 
                   (text_x, text_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Insert the annotated ROI back into the frame
    frame[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_roi
    
    # Create a grid of all the processed images for debugging
    top_row = np.hstack([roi, enhanced])
    bottom_row = np.hstack([masked_roi, edge_enhanced])
    processed_grid = np.vstack([top_row, bottom_row])
    
    # Resize the grid to fit the display
    processed_grid = cv2.resize(processed_grid, (ROI_SIZE * 2, ROI_SIZE * 2))
    
    # Add labels to the grid
    cv2.putText(processed_grid, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(processed_grid, "Enhanced", (ROI_SIZE + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(processed_grid, "Skin Masked", (10, ROI_SIZE + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(processed_grid, "Edge Enhanced", (ROI_SIZE + 10, ROI_SIZE + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Show FPS on the main frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the main frame and the processed grid
    cv2.imshow("Sign Language Detection", frame)
    cv2.imshow("Preprocessing Steps", processed_grid)
    cv2.imshow("Thresholded", thresholded)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----- CLEANUP -----
cap.release()
cv2.destroyAllWindows()
print("Application closed.")