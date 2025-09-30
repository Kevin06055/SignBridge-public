# SignBridge Backend

A production-ready Flask backend for the SignBridge Indian Sign Language platform. This unified backend integrates text-to-sign conversion and real-time sign detection capabilities.

## Features

- **Text-to-Sign Conversion**: Convert text to sign language videos with customizable options
- **Real-time Sign Detection**: YOLO-based sign language detection with preprocessing
- **Modular Architecture**: Clean separation of concerns with Flask blueprints
- **Production Ready**: Comprehensive error handling, logging, and configuration management
- **Firebase Compatible**: Ready for deployment on Firebase Functions
- **CORS Enabled**: Full support for frontend integration

## Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Installation

1. **Clone the repository and navigate to the backend:**
   ```bash
   cd SignBridge/BackEnd
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Copy `.env` and update values as needed:
   ```bash
   # Windows
   copy .env .env.local
   
   # macOS/Linux
   cp .env .env.local
   ```

5. **Ensure required directories exist:**
   The application will create these automatically, but you can verify:
   - `TextToSignPipeline/sign_images/` - Contains A.png, B.png, etc.
   - `TextToSignPipeline/output_videos/` - For generated videos
   - `logs/` - For application logs

6. **Run the application:**
   ```bash
   python run.py
   ```

   The server will start at `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/health` - Application health status
- **GET** `/api` - API information and available endpoints

### Text-to-Sign
- **POST** `/api/v1/text-to-sign/convert` - Start text-to-sign conversion
- **GET** `/api/v1/text-to-sign/status/{task_id}` - Get conversion status
- **GET** `/api/v1/text-to-sign/download/{task_id}` - Download generated video
- **GET** `/api/v1/text-to-sign/available-letters` - Get available sign letters
- **GET** `/api/v1/text-to-sign/health` - Service health check

### Sign Detection
- **POST** `/api/v1/sign-detection/detect` - Detect signs in image
- **POST** `/api/v1/sign-detection/detect-realtime` - Real-time detection (optimized)
- **GET** `/api/v1/sign-detection/model/info` - YOLO model information
- **POST** `/api/v1/sign-detection/reset-history` - Reset prediction history
- **GET** `/api/v1/sign-detection/configuration` - Get detection configuration
- **GET** `/api/v1/sign-detection/health` - Service health check

### Speech-to-Sign
- **POST** `/api/v1/speech-to-sign/convert` - Convert speech text to sign language
- **GET** `/api/v1/speech-to-sign/status/{task_id}` - Get conversion status
- **GET** `/api/v1/speech-to-sign/download/{task_id}` - Download generated video
- **GET** `/api/v1/speech-to-sign/available-letters` - Get available sign letters
- **POST** `/api/v1/speech-to-sign/preview` - Preview conversion coverage
- **GET** `/api/v1/speech-to-sign/health` - Service health check

## API Usage Examples

### Text-to-Sign Conversion

```bash
# Start conversion
curl -X POST http://localhost:5000/api/v1/text-to-sign/convert \
  -H "Content-Type: application/json" \
  -d '{
    "text": "HELLO WORLD",
    "options": {
      "summarize": false,
      "frame_duration": 1.0,
      "video_fps": 1,
      "filename": "hello_world.mp4"
    }
  }'

# Check status
curl http://localhost:5000/api/v1/text-to-sign/status/{task_id}

# Download video (when ready)
curl -O http://localhost:5000/api/v1/text-to-sign/download/{task_id}
```

### Sign Detection

```bash
# Detect signs in image
curl -X POST http://localhost:5000/api/v1/sign-detection/detect \
  -H "Content-Type: application/json" \
  -d '{
    "frame": "base64_encoded_image_data",
    "confidence": 0.5,
    "smoothing": true
  }'
```

### Speech-to-Sign Conversion

```bash
# Convert speech text to sign language
curl -X POST http://localhost:5000/api/v1/speech-to-sign/convert \
  -H "Content-Type: application/json" \
  -d '{
    "text": "HELLO WORLD",
    "frame_duration": 1.0,
    "video_fps": 1,
    "filename": "speech_hello_world.mp4"
  }'

# Preview conversion coverage
curl -X POST http://localhost:5000/api/v1/speech-to-sign/preview \
  -H "Content-Type: application/json" \
  -d '{"text": "HELLO WORLD"}'

# Check status
curl http://localhost:5000/api/v1/speech-to-sign/status/{task_id}
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Application
FLASK_ENV=development
FLASK_DEBUG=true
SECRET_KEY=your-secret-key
HOST=0.0.0.0
PORT=5000

# Frontend URLs
FRONTEND_URL=https://your-frontend.com

# YOLO Configuration
YOLO_CONFIDENCE=0.5
YOLO_MODEL_PATH=fine_tuned.pt
YOLO_FALLBACK_MODEL=yolov8n.pt

# Logging
LOG_LEVEL=INFO

# API Keys
API_KEYS=default-api-key,production-key

# Firebase (for deployment)
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_CREDENTIALS_PATH=path/to/credentials.json
```

### Directory Structure

```
BackEnd/
├── app.py                 # Main Flask application
├── config.py             # Configuration management
├── run.py                # Application startup script
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── blueprints/           # API endpoints
│   ├── text_to_sign.py  # Text-to-sign endpoints
│   └── sign_detection.py # Sign detection endpoints
├── utils/                # Utilities
│   └── logger.py        # Logging configuration
├── tests/               # Test suite
├── logs/                # Application logs
├── TextToSignPipeline/  # Text-to-sign resources
│   ├── sign_images/    # Sign language images (A.png, B.png, etc.)
│   └── output_videos/  # Generated videos
└── uploads/            # Temporary file uploads
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-flask

# Run all tests
pytest

# Run specific test file
pytest tests/test_app.py

# Run with coverage
pytest --cov=. tests/
```

## Production Deployment

### Option 1: Traditional Server

1. **Set production environment variables:**
   ```env
   FLASK_ENV=production
   FLASK_DEBUG=false
   SECRET_KEY=your-production-secret-key
   ```

2. **Use a WSGI server like Gunicorn:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Option 2: Firebase Functions

1. **Install Firebase CLI:**
   ```bash
   npm install -g firebase-tools
   ```

2. **Initialize Firebase:**
   ```bash
   firebase init functions
   ```

3. **Deploy:**
   ```bash
   firebase deploy --only functions
   ```

### Option 3: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "run.py"]
```

## Frontend Integration

The React frontend is configured to work with this backend. Update the frontend's `.env`:

```env
# Development
VITE_API_URL=http://localhost:5000

# Production
VITE_API_URL=https://your-backend-domain.com
```

## Troubleshooting

### Common Issues

1. **YOLO Model Not Found:**
   - Ensure `fine_tuned.pt` exists in the project root
   - The system will fallback to `yolov8n.pt` if available
   - Check logs for model loading errors

2. **Sign Images Missing:**
   - Verify `TextToSignPipeline/sign_images/` contains A.png, B.png, etc.
   - Images should be in PNG or JPG format

3. **CORS Issues:**
   - Check `CORS_ORIGINS` in config.py
   - Ensure frontend URL is included

4. **Permission Errors:**
   - Ensure write permissions for `logs/` and `TextToSignPipeline/output_videos/`

### Logs

Check application logs in `logs/signbridge.log` for detailed error information.

## Development

### Adding New Features

1. **Create a new blueprint:**
   ```python
   # blueprints/new_feature.py
   from flask import Blueprint
   
   new_feature_bp = Blueprint('new_feature', __name__)
   
   @new_feature_bp.route('/endpoint')
   def endpoint():
       return {'message': 'Hello from new feature'}
   ```

2. **Register the blueprint:**
   ```python
   # app.py
   from blueprints.new_feature import new_feature_bp
   app.register_blueprint(new_feature_bp, url_prefix='/api/v1/new-feature')
   ```

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where possible
- Add docstrings for all functions and classes
- Include comprehensive error handling

## License

This project is part of the SignBridge platform. See the main repository for license information.

## Support

For issues and questions:
1. Check the logs in `logs/signbridge.log`
2. Verify configuration in `.env`
3. Run health checks: `http://localhost:5000/health`
4. Check individual service health endpoints