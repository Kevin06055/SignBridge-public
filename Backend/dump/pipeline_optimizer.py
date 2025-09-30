import cv2
import numpy as np
import os
import time
import argparse
import itertools
from ultralytics import YOLO
from collections import defaultdict
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Optimize preprocessing pipeline for sign language detection')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to YOLO model file')
    parser.add_argument('--test-dir', type=str, default='data/test/images', help='Directory with test images')
    parser.add_argument('--output', type=str, default='pipeline_optimization', help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--samples', type=int, default=10, help='Number of sample images to test')
    parser.add_argument('--visualize', action='store_true', help='Save visualization of best pipeline')
    return parser.parse_args()

class PreprocessingOptimizer:
    def __init__(self, model_path, conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Define preprocessing techniques
        self.techniques = {
            'original': self.original,
            'bilateral': self.bilateral_filter,
            'contrast': self.contrast_enhancement,
            'skin': self.skin_segmentation,
            'edge': self.edge_detection,
            'threshold': self.adaptive_threshold,
            'color_filter': self.color_filtering
        }
        
        # Default class names
        self.class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Load class names from file if available
        if os.path.exists('data/classes.txt'):
            with open('data/classes.txt', 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
    
    def original(self, image):
        """Return the original image"""
        return image.copy()
    
    def bilateral_filter(self, image):
        """Apply bilateral filter for edge-preserving smoothing"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def contrast_enhancement(self, image):
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
    
    def skin_segmentation(self, image):
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
        
        return skin_segmented
    
    def edge_detection(self, image):
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
    
    def adaptive_threshold(self, image):
        """Apply adaptive thresholding for varying lighting conditions"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Convert back to BGR for consistency
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return thresh_bgr
    
    def color_filtering(self, image):
        """Filter specific color ranges common in sign language datasets"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Filter for skin and bright colored markers if any
        lower_bound = np.array([0, 30, 60])
        upper_bound = np.array([30, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        color_filtered = cv2.bitwise_and(image, image, mask=mask)
        
        return color_filtered
    
    def apply_pipeline(self, image, pipeline):
        """Apply a sequence of preprocessing techniques"""
        result = image.copy()
        
        for technique in pipeline:
            result = self.techniques[technique](result)
        
        return result
    
    def generate_all_pipelines(self, max_length=3):
        """Generate all possible preprocessing pipelines up to max_length"""
        all_pipelines = []
        
        # Always include the original
        all_pipelines.append(('original',))
        
        # Generate combinations
        techniques = list(self.techniques.keys())
        techniques.remove('original')  # Remove original as it will be the first step
        
        for length in range(1, max_length + 1):
            for combo in itertools.combinations(techniques, length):
                # Start with original, then apply techniques
                pipeline = ('original',) + combo
                all_pipelines.append(pipeline)
                
                # Also try without starting with original
                if length > 1:
                    all_pipelines.append(combo)
        
        return all_pipelines
    
    def evaluate_pipeline(self, image, pipeline, ground_truth=None):
        """Evaluate a pipeline on an image"""
        # Apply pipeline
        processed = self.apply_pipeline(image, pipeline)
        
        # Run inference
        results = self.model(processed)[0]
        
        # Extract detections above confidence threshold
        detections = []
        for det in results.boxes:
            conf = det.conf.item()
            if conf > self.conf_threshold:
                cls_id = int(det.cls.item())
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                detections.append((cls_id, conf, (x1, y1, x2, y2)))
        
        # Get highest confidence detection
        if detections:
            best_detection = max(detections, key=lambda x: x[1])
            return {
                'detected': True,
                'class_id': best_detection[0],
                'confidence': best_detection[1],
                'bbox': best_detection[2],
                'num_detections': len(detections)
            }
        else:
            return {
                'detected': False,
                'class_id': None,
                'confidence': 0,
                'bbox': None,
                'num_detections': 0
            }
    
    def create_pipeline_visualization(self, image, pipeline):
        """Create a visualization of the preprocessing pipeline"""
        steps = []
        current = image.copy()
        
        # Apply each step and save intermediate results
        steps.append(("Original", current.copy()))
        
        for technique in pipeline:
            if technique == 'original' and len(steps) > 0:
                continue  # Skip if we already have the original
            
            current = self.techniques[technique](current)
            steps.append((technique.capitalize(), current.copy()))
        
        # Create a grid of all steps
        h, w = image.shape[:2]
        num_steps = len(steps)
        
        if num_steps <= 4:
            # Single row
            grid_width = w * num_steps
            grid_height = h
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            for i, (name, img) in enumerate(steps):
                grid[0:h, i*w:(i+1)*w] = img
                
                # Add label
                cv2.putText(grid, name, (i*w + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Two rows
            row_length = (num_steps + 1) // 2
            grid_width = w * row_length
            grid_height = h * 2
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            for i, (name, img) in enumerate(steps):
                row = i // row_length
                col = i % row_length
                
                grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
                
                # Add label
                cv2.putText(grid, name, (col*w + 10, row*h + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return grid

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize optimizer
    optimizer = PreprocessingOptimizer(args.model, args.conf)
    
    # Get test images
    image_files = [f for f in os.listdir(args.test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {args.test_dir}")
        return
    
    # Select sample images
    num_samples = min(args.samples, len(image_files))
    sample_files = image_files[:num_samples]
    
    print(f"Testing {num_samples} sample images...")
    
    # Generate all possible pipelines
    pipelines = optimizer.generate_all_pipelines(max_length=3)
    print(f"Testing {len(pipelines)} different preprocessing pipelines")
    
    # Results storage
    pipeline_results = defaultdict(list)
    
    # Process each sample
    for i, file_name in enumerate(sample_files):
        print(f"Processing sample {i+1}/{num_samples}: {file_name}")
        
        # Load image
        image_path = os.path.join(args.test_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue
        
        # Evaluate each pipeline
        for pipeline in pipelines:
            result = optimizer.evaluate_pipeline(image, pipeline)
            
            # Store results
            pipeline_str = ' → '.join(pipeline)
            pipeline_results[pipeline_str].append({
                'file': file_name,
                'detected': result['detected'],
                'class_id': result['class_id'],
                'confidence': result['confidence'],
                'num_detections': result['num_detections']
            })
    
    # Calculate aggregate metrics for each pipeline
    pipeline_metrics = {}
    
    for pipeline_str, results in pipeline_results.items():
        # Count detections
        num_detected = sum(1 for r in results if r['detected'])
        
        # Calculate average confidence
        avg_conf = sum(r['confidence'] for r in results) / len(results) if results else 0
        
        # Calculate average number of detections
        avg_detections = sum(r['num_detections'] for r in results) / len(results) if results else 0
        
        pipeline_metrics[pipeline_str] = {
            'detection_rate': num_detected / len(results) if results else 0,
            'avg_confidence': avg_conf,
            'avg_detections': avg_detections,
            'pipeline': pipeline_str.split(' → ')
        }
    
    # Sort pipelines by detection rate, then by average confidence
    sorted_pipelines = sorted(
        pipeline_metrics.items(), 
        key=lambda x: (x[1]['detection_rate'], x[1]['avg_confidence']), 
        reverse=True
    )
    
    # Print top 5 pipelines
    print("\nTop 5 preprocessing pipelines:")
    for i, (pipeline_str, metrics) in enumerate(sorted_pipelines[:5]):
        print(f"{i+1}. {pipeline_str}")
        print(f"   Detection Rate: {metrics['detection_rate']:.2%}")
        print(f"   Avg Confidence: {metrics['avg_confidence']:.4f}")
        print(f"   Avg Detections: {metrics['avg_detections']:.2f}")
    
    # Get the best pipeline
    best_pipeline_str, best_metrics = sorted_pipelines[0]
    best_pipeline = best_metrics['pipeline']
    
    # Save detailed results to JSON
    results_file = os.path.join(args.output, 'pipeline_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'top_pipelines': [
                {
                    'pipeline': p,
                    'metrics': m
                } for p, m in sorted_pipelines[:10]
            ],
            'all_metrics': pipeline_metrics,
            'test_parameters': {
                'model': args.model,
                'num_samples': num_samples,
                'confidence_threshold': args.conf
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")
    
    # Generate code for the best pipeline
    code_file = os.path.join(args.output, 'best_pipeline.py')
    with open(code_file, 'w') as f:
        f.write(f'''import cv2
import numpy as np

def apply_best_preprocessing(image):
    """
    Apply the optimized preprocessing pipeline: {best_pipeline_str}
    
    Args:
        image: BGR image to preprocess
        
    Returns:
        Preprocessed image ready for model inference
    """
    # Make a copy of the input image
    result = image.copy()
    
''')
        # Generate code for each step
        for step in best_pipeline:
            if step == 'original':
                f.write("    # Original image (no preprocessing)\n")
                f.write("    # This step is a no-op\n\n")
            elif step == 'bilateral':
                f.write("    # Apply bilateral filter (edge-preserving smoothing)\n")
                f.write("    result = cv2.bilateralFilter(result, 9, 75, 75)\n\n")
            elif step == 'contrast':
                f.write("    # Apply contrast enhancement using CLAHE\n")
                f.write("    # Convert to LAB color space\n")
                f.write("    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)\n")
                f.write("    # Split channels\n")
                f.write("    l, a, b = cv2.split(lab)\n")
                f.write("    # Apply CLAHE to L channel\n")
                f.write("    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))\n")
                f.write("    cl = clahe.apply(l)\n")
                f.write("    # Merge channels\n")
                f.write("    merged = cv2.merge((cl, a, b))\n")
                f.write("    # Convert back to BGR\n")
                f.write("    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)\n\n")
            elif step == 'skin':
                f.write("    # Apply skin segmentation\n")
                f.write("    # Convert to HSV\n")
                f.write("    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)\n")
                f.write("    # Define skin color ranges\n")
                f.write("    lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)\n")
                f.write("    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)\n")
                f.write("    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)\n")
                f.write("    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)\n")
                f.write("    # Create masks\n")
                f.write("    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)\n")
                f.write("    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)\n")
                f.write("    mask = cv2.bitwise_or(mask1, mask2)\n")
                f.write("    # Apply morphology\n")
                f.write("    kernel = np.ones((5, 5), np.uint8)\n")
                f.write("    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)\n")
                f.write("    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)\n")
                f.write("    # Apply mask\n")
                f.write("    result = cv2.bitwise_and(result, result, mask=mask)\n\n")
            elif step == 'edge':
                f.write("    # Apply edge detection\n")
                f.write("    # Convert to grayscale\n")
                f.write("    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n")
                f.write("    # Apply Gaussian blur\n")
                f.write("    blurred = cv2.GaussianBlur(gray, (5, 5), 0)\n")
                f.write("    # Apply Canny edge detection\n")
                f.write("    edges = cv2.Canny(blurred, 50, 150)\n")
                f.write("    # Convert back to BGR\n")
                f.write("    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)\n\n")
            elif step == 'threshold':
                f.write("    # Apply adaptive thresholding\n")
                f.write("    # Convert to grayscale\n")
                f.write("    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n")
                f.write("    # Apply adaptive threshold\n")
                f.write("    thresh = cv2.adaptiveThreshold(\n")
                f.write("        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\n") 
                f.write("        cv2.THRESH_BINARY_INV, 11, 2\n")
                f.write("    )\n")
                f.write("    # Convert back to BGR\n")
                f.write("    result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)\n\n")
            elif step == 'color_filter':
                f.write("    # Apply color filtering\n")
                f.write("    # Convert to HSV\n")
                f.write("    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)\n")
                f.write("    # Define color range\n")
                f.write("    lower_bound = np.array([0, 30, 60])\n")
                f.write("    upper_bound = np.array([30, 255, 255])\n")
                f.write("    # Create mask\n")
                f.write("    mask = cv2.inRange(hsv, lower_bound, upper_bound)\n")
                f.write("    # Apply mask\n")
                f.write("    result = cv2.bitwise_and(result, result, mask=mask)\n\n")
        
        f.write("    return result\n")
    
    print(f"Best pipeline code saved to {code_file}")
    
    # Visualize the best pipeline on a sample image if requested
    if args.visualize:
        for i, file_name in enumerate(sample_files[:3]):  # Use first 3 samples
            # Load image
            image_path = os.path.join(args.test_dir, file_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Create visualization
            vis = optimizer.create_pipeline_visualization(image, best_pipeline)
            
            # Save visualization
            vis_path = os.path.join(args.output, f"pipeline_vis_{i+1}.jpg")
            cv2.imwrite(vis_path, vis)
            
            print(f"Visualization saved to {vis_path}")

if __name__ == "__main__":
    main()