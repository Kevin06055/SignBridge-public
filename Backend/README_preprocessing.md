# SignBridge Preprocessing & Inference Tools

This repository contains tools for optimizing sign language detection using various preprocessing techniques.

## Overview

These scripts allow you to:

1. Test different preprocessing techniques on sign language images
2. Analyze which preprocessing techniques work best for your model
3. Optimize the preprocessing pipeline for best detection results
4. Run real-time inference with the optimized preprocessing pipeline

## Files

- `preprocessing_tester.py`: Test various preprocessing techniques on individual images
- `preprocessing_analyzer.py`: Interactive tool to analyze different preprocessing approaches in real-time
- `pipeline_optimizer.py`: Find the optimal preprocessing pipeline using your test dataset
- `realtime_inference.py`: Run real-time sign language detection with optimized preprocessing
- `inference_enhanced.py`: Comprehensive inference script with multiple preprocessing techniques

## Usage Instructions

### 1. Test Preprocessing Techniques

This script allows you to test different preprocessing techniques on single images or a batch of test images:

```bash
python preprocessing_tester.py --model best.pt --image data/test/images/sample.jpg --conf 0.25
```

Or test on multiple images from a directory:

```bash
python preprocessing_tester.py --model best.pt --test-dir data/test/images --conf 0.25
```

### 2. Analyze Preprocessing Techniques

This script provides an interactive tool to analyze different preprocessing approaches in real-time:

```bash
python preprocessing_analyzer.py --model best.pt --input 0 --show-grid --save-results
```

Options:
- `--input`: Camera device (default 0) or path to video file
- `--show-grid`: Show visualization grid of preprocessing steps
- `--save-results`: Save preprocessing analysis images when pressing 's'

### 3. Optimize Preprocessing Pipeline

This script finds the optimal preprocessing pipeline by testing combinations on your dataset:

```bash
python pipeline_optimizer.py --model best.pt --test-dir data/test/images --samples 10 --visualize
```

Options:
- `--samples`: Number of sample images to test (default 10)
- `--visualize`: Save visualization of the best pipeline
- `--conf`: Confidence threshold (default 0.3)

The script generates:
- `pipeline_results.json`: Detailed results of all tested pipelines
- `best_pipeline.py`: Ready-to-use code for the best pipeline
- Visualization images of the best pipeline applied to samples

### 4. Run Real-time Inference

Run real-time sign language detection with the optimized preprocessing:

```bash
python realtime_inference.py --model best.pt --source 0 --show-preprocessing
```

Options:
- `--source`: Camera device number (default 0)
- `--show-preprocessing`: Show preprocessing steps
- `--record`: Path to save video recording
- `--smooth`: Frames for temporal smoothing (default 5, 0 to disable)

### 5. Enhanced Inference

Use the comprehensive inference script with multiple preprocessing techniques:

```bash
python inference_enhanced.py
```

This script uses a combination of preprocessing techniques and ensemble detection for robust sign language recognition.

## Tips for Best Results

1. **Lighting**: Ensure good, consistent lighting on the hand for better detection
2. **Background**: Use a simple background to reduce false detections
3. **Hand Placement**: Keep your hand within the green ROI box
4. **Temporal Smoothing**: For real-time applications, use temporal smoothing to reduce jitter
5. **Pipeline Optimization**: Run `pipeline_optimizer.py` on your specific dataset to find the best preprocessing for your environment

## Dependencies

- Python 3.6+
- OpenCV 4.5+
- NumPy
- Ultralytics YOLO
- A trained sign language detection model (default: `best.pt`)

## Installation

```bash
pip install opencv-python numpy ultralytics
```

## Acknowledgments

This project uses the YOLO (You Only Look Once) object detection architecture for sign language detection.