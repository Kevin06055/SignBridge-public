import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test various preprocessing techniques for sign language detection')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to YOLO model file')
    parser.add_argument('--image', type=str, default='', help='Path to single test image (optional)')
    parser.add_argument('--test-dir', type=str, default='data/test/images', help='Directory with test images')
    parser.add_argument('--output', type=str, default='preprocessing_results', help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    return parser.parse_args()

def apply_skin_segmentation(image):
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
    
    return skin_segmented, mask

def apply_edge_enhancement(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Combine original with edges
    edge_enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edge_enhanced[:, :, 0] = edges  # Apply edges to blue channel
    
    return edge_enhanced

def apply_contrast_enhancement(image):
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

def apply_adaptive_thresholding(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Convert back to BGR
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    return thresh_bgr

def apply_histogram_equalization(image):
    # Convert to YUV
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Equalize the histogram of the Y channel
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    
    # Convert back to BGR
    hist_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    return hist_eq

def apply_color_filtering(image):
    # Filter specific color ranges that are common in sign language datasets
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Filter for skin and bright colored markers if any
    lower_bound = np.array([0, 30, 60])
    upper_bound = np.array([30, 255, 255])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    color_filtered = cv2.bitwise_and(image, image, mask=mask)
    
    return color_filtered

def apply_bilateral_filter(image):
    # Apply bilateral filter for edge-preserving smoothing
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    return bilateral

def process_image_with_all_techniques(image):
    # Apply all preprocessing techniques
    skin_segmented, skin_mask = apply_skin_segmentation(image)
    edge_enhanced = apply_edge_enhancement(image)
    contrast_enhanced = apply_contrast_enhancement(image)
    thresh = apply_adaptive_thresholding(image)
    hist_eq = apply_histogram_equalization(image)
    color_filtered = apply_color_filtering(image)
    bilateral = apply_bilateral_filter(image)
    
    # Create a result dictionary
    results = {
        'original': image,
        'skin_segmented': skin_segmented,
        'skin_mask': cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR),
        'edge_enhanced': edge_enhanced,
        'contrast_enhanced': contrast_enhanced,
        'threshold': thresh,
        'hist_eq': hist_eq,
        'color_filtered': color_filtered,
        'bilateral': bilateral
    }
    
    return results

def create_visualization_grid(processed_results):
    # Create a 3x3 grid of processed images
    top_row = np.hstack([processed_results['original'], processed_results['skin_segmented'], processed_results['skin_mask']])
    middle_row = np.hstack([processed_results['edge_enhanced'], processed_results['contrast_enhanced'], processed_results['threshold']])
    bottom_row = np.hstack([processed_results['hist_eq'], processed_results['color_filtered'], processed_results['bilateral']])
    
    grid = np.vstack([top_row, middle_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = processed_results['original'].shape[:2]
    
    labels = [
        ('Original', 0, 0), ('Skin Segmented', w, 0), ('Skin Mask', 2*w, 0),
        ('Edge Enhanced', 0, h), ('Contrast Enhanced', w, h), ('Threshold', 2*w, h),
        ('Hist Equalization', 0, 2*h), ('Color Filtered', w, 2*h), ('Bilateral', 2*w, 2*h)
    ]
    
    for label, x, y in labels:
        cv2.putText(grid, label, (x + 10, y + 30), font, 0.7, (0, 255, 0), 2)
    
    return grid

def run_model_on_all_versions(model, processed_results, conf_threshold=0.25):
    predictions = {}
    
    for name, img in processed_results.items():
        # Skip the skin mask which is just for visualization
        if name == 'skin_mask':
            continue
            
        # Run inference
        results = model(img)[0]
        
        # Extract detections above confidence threshold
        detections = []
        for det in results.boxes:
            conf = det.conf.item()
            if conf > conf_threshold:
                cls_id = int(det.cls.item())
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                detections.append((cls_id, conf, (x1, y1, x2, y2)))
        
        predictions[name] = detections
    
    return predictions

def draw_predictions(processed_results, predictions, class_names):
    result_images = {}
    
    for name, img in processed_results.items():
        # Skip the skin mask
        if name == 'skin_mask':
            continue
            
        # Create a copy to draw on
        result_img = img.copy()
        
        # Draw detections
        for cls_id, conf, (x1, y1, x2, y2) in predictions.get(name, []):
            # Get class name
            class_label = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            
            # Draw bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_label} {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        result_images[name] = result_img
    
    return result_images

def create_prediction_grid(result_images):
    # Create a 3x3 grid of result images
    top_row = np.hstack([result_images['original'], result_images['skin_segmented'], result_images.get('placeholder1', np.zeros_like(result_images['original']))])
    middle_row = np.hstack([result_images['edge_enhanced'], result_images['contrast_enhanced'], result_images['threshold']])
    bottom_row = np.hstack([result_images['hist_eq'], result_images['color_filtered'], result_images['bilateral']])
    
    grid = np.vstack([top_row, middle_row, bottom_row])
    
    # Add labels and detection counts
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = result_images['original'].shape[:2]
    
    labels = [
        ('Original', 0, 0), ('Skin Segmented', w, 0), ('', 2*w, 0),
        ('Edge Enhanced', 0, h), ('Contrast Enhanced', w, h), ('Threshold', 2*w, h),
        ('Hist Equalization', 0, 2*h), ('Color Filtered', w, 2*h), ('Bilateral', 2*w, 2*h)
    ]
    
    for label, x, y in labels:
        cv2.putText(grid, label, (x + 10, y + 30), font, 0.7, (0, 255, 0), 2)
    
    return grid

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load the YOLO model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Load class names from data.yaml or use default
    class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    if args.image:
        # Process a single image
        print(f"Processing image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        
        # Process the image with all techniques
        processed_results = process_image_with_all_techniques(image)
        
        # Create visualization grid
        grid = create_visualization_grid(processed_results)
        
        # Run model on all versions
        predictions = run_model_on_all_versions(model, processed_results, args.conf)
        
        # Draw predictions
        result_images = draw_predictions(processed_results, predictions, class_names)
        
        # Create prediction grid
        prediction_grid = create_prediction_grid(result_images)
        
        # Display results
        cv2.imshow("Preprocessing Techniques", grid)
        cv2.imshow("Detection Results", prediction_grid)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        cv2.imwrite(os.path.join(args.output, f"{base_name}_preprocessing.jpg"), grid)
        cv2.imwrite(os.path.join(args.output, f"{base_name}_detections.jpg"), prediction_grid)
        
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Process test directory
        print(f"Processing images in directory: {args.test_dir}")
        
        image_files = [f for f in os.listdir(args.test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"No images found in {args.test_dir}")
            return
        
        # Process a few sample images
        sample_count = min(5, len(image_files))
        for i in range(sample_count):
            image_path = os.path.join(args.test_dir, image_files[i])
            print(f"Processing image {i+1}/{sample_count}: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {image_path}")
                continue
            
            # Process the image with all techniques
            processed_results = process_image_with_all_techniques(image)
            
            # Create visualization grid
            grid = create_visualization_grid(processed_results)
            
            # Run model on all versions
            predictions = run_model_on_all_versions(model, processed_results, args.conf)
            
            # Draw predictions
            result_images = draw_predictions(processed_results, predictions, class_names)
            
            # Create prediction grid
            prediction_grid = create_prediction_grid(result_images)
            
            # Save results
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(args.output, f"{base_name}_preprocessing.jpg"), grid)
            cv2.imwrite(os.path.join(args.output, f"{base_name}_detections.jpg"), prediction_grid)
            
            # Display results (optional for batch processing)
            cv2.imshow("Preprocessing Techniques", grid)
            cv2.imshow("Detection Results", prediction_grid)
            
            key = cv2.waitKey(1000)  # Wait 1 second between images or press a key to skip
            if key != -1:
                break
        
        cv2.destroyAllWindows()
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()