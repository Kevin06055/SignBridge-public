# SignBridge - Enhanced Indian Sign Language Platform

A comprehensive platform for Indian Sign Language recognition, speech-to-sign conversion, text summarization, Braille conversion, and course material management. The system uses a modular architecture with separate Firebase accounts for hosting and API deployment.

## 🏗️ Architecture Overview

```
SignBridge/
├── sign-talk-pal/              # Frontend React App (Firebase Hosting Account)
├── BackEnd/                    # Backend API Services (Firebase Functions Account)
│   ├── functions/              # Firebase Functions entry point
│   ├── shared/                 # Shared utilities and configuration
│   ├── SignDetectionPipeline/  # Real-time sign language detection
│   ├── SpeechToSignPipeline/   # Speech-to-sign conversion
│   ├── TextSummarizationPipeline/ # Content summarization
│   ├── BrailleConversionPipeline/  # Text-to-Braille conversion
│   └── CourseMaterialPipeline/ # Course material management
└── docs/                       # Documentation
```

## 🚀 Quick Start

## Setup Instructions

### Prerequisites

- Python 3.8+ with pip
- Node.js and npm/yarn
- Webcam

### Backend Setup

1. Install the required Python packages:

```bash
pip install flask flask-cors opencv-python numpy ultralytics
```

2. Start the API server:

```bash
python BackEnd\run.py
```

The server will run on http://localhost:5000 by default.

### Frontend Setup

1. Navigate to the sign-talk-pal directory:

```bash
cd sign-talk-pal
```

2. Install dependencies:

```bash
npm install
# or
yarn install
```

3. Start the development server:

```bash
npm run dev
# or
yarn dev
```

The frontend will be available at http://localhost:5173 (or another port if 5173 is in use).

## Usage

1. Open the frontend application in your browser
2. Navigate to the "Sign to Text" page
3. Click the "Start Recognition" button to activate your webcam
4. Position your hand in the center of the camera view
5. Make sign language gestures, and the system will detect and display the corresponding letters/numbers
6. Click "Stop Recognition" to end the session

## Available Scripts

### Backend

- `realtime_inference.py` - The main script for sign language detection, can run in standalone or API mode
- `start_api_server.py` - A simple script to start the API server
- `sign_detection_api.py` - A dedicated API server for sign language detection
- `realtime_inference_backup.py` - A backup of the original inference script

### Frontend

The frontend is a React/TypeScript application with TailwindCSS for styling. The main component for sign language detection is `SignToText.tsx`.

## Command Line Options

### API Server

```bash
python start_api_server.py --port 5000 --model best.pt
```

- `--port` - Port for the API server (default: 5000)
- `--model` - Path to the YOLO model file (default: best.pt)

### Standalone Mode

```bash
python realtime_inference.py --source 0 --conf 0.5 --show-preprocessing --smooth 5
```

- `--source` - Camera device number (default: 0)
- `--conf` - Confidence threshold (default: 0.5)
- `--show-preprocessing` - Show preprocessing steps
- `--record` - Path to save video recording
- `--smooth` - Frames for temporal smoothing (default: 5, 0 to disable)
- `--api` - Run as API server
- `--port` - Port for API server (default: 5000)

## Technical Details

### Preprocessing Pipeline

The system uses an optimized preprocessing pipeline for better sign detection:

1. Bilateral filtering for noise reduction while preserving edges
2. Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Skin segmentation using HSV color space

### Temporal Smoothing

To reduce flickering and improve stability, the system applies temporal smoothing:

1. Maintains a history of recent detections
2. Uses a majority voting system to determine the most stable prediction
3. Only changes the displayed sign when a new sign is detected consistently

### API Endpoints

- `GET /api/status` - Check if the API is running and model is loaded
- `POST /api/detect` - Send a base64-encoded image frame for sign detection

## Troubleshooting

1. **Camera access denied**: Make sure your browser has permission to access your webcam
2. **API connection failed**: Ensure the API server is running on the correct port
3. **Low detection accuracy**: Try adjusting lighting conditions and position your hand in the center of the frame
4. **Model not loading**: Check that the YOLO model file exists at the specified path

## License

This project is licensed under the MIT License - see the LICENSE file for details.
