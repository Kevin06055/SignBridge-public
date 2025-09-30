# SignBridge Streamlit Demo

A comprehensive interactive demonstration of all SignBridge features built with Streamlit.

## 🌟 Features

### 🎤 Speech-to-Sign Conversion
- Convert spoken words into sign language videos
- Real-time processing simulation
- Customizable video settings (FPS, frame duration)
- Download generated videos

### 📝 Text-to-Sign Language 
- Transform written text into animated sign sequences
- Support for A-Z alphabet signs
- Interactive video generation
- Live text preview

### 👁️ Sign Language Detection
- Upload images for sign language analysis
- AI-powered gesture recognition
- Confidence score reporting
- Visual detection overlay

### 📚 Interactive Learning Module
- Learn sign language alphabet step by step
- Practice mode with random challenges
- Progress tracking
- Visual learning aids

### ⚡ Real-time Processing
- Live text-to-sign preview
- Instant feedback as you type
- Real-time sign image display

### 📊 Analytics & Statistics
- Processing history tracking
- Success rate monitoring
- System performance metrics
- Session analytics

## 🚀 Quick Start

### Option 1: Windows Batch Script
```bash
# Double-click to run
run_streamlit_demo.bat
```

### Option 2: Manual Launch
```bash
# Install dependencies
pip install -r streamlit_requirements.txt

# Run the demo
streamlit run streamlit_demo.py
```

### Option 3: Custom Port
```bash
streamlit run streamlit_demo.py --server.port 8502
```

## 📋 Requirements

- Python 3.8+
- Sign images in `BackEnd/TextToSignPipeline/sign_images/` (A.png through Z.png)
- 2GB+ free disk space for video generation
- Modern web browser

## 🎯 Demo Features

### Video Generation
- Creates MP4 videos from text input
- Customizable frame rates and durations
- Text overlay options
- Instant download capability

### Sign Detection (Mock)
- Simulates YOLO-based sign detection
- Visual bounding box overlays
- Confidence score display
- Multiple sign recognition

### Interactive Learning
- A-Z alphabet learning module
- Visual sign reference
- Practice challenges
- Progress tracking

## 🔧 Configuration

The demo automatically detects available sign images and adapts features accordingly:

- **Full alphabet (A-Z)**: All features enabled
- **Partial alphabet**: Features work with available letters
- **No sign images**: Demo mode with mock data

## 📁 File Structure

```
SignBridge/
├── streamlit_demo.py           # Main Streamlit application
├── streamlit_requirements.txt  # Python dependencies
├── run_streamlit_demo.bat     # Windows launch script
├── STREAMLIT_README.md        # This file
└── BackEnd/
    └── TextToSignPipeline/
        └── sign_images/       # Sign language images (A.png - Z.png)
```

## 🎨 User Interface

### Navigation Sidebar
- **🏠 Home & Overview**: System introduction and statistics
- **🎤 Speech-to-Sign**: Speech conversion simulation
- **📝 Text-to-Sign**: Text input to video generation  
- **👁️ Sign Detection**: Image upload and analysis
- **📚 Learning Module**: Interactive alphabet learning
- **⚡ Real-time Processing**: Live text preview
- **📊 Analytics & Stats**: Session history and metrics

### Interactive Elements
- File uploaders for images
- Text input areas
- Sliders for video settings
- Buttons for processing
- Download links for generated content

## 🛠️ Technical Details

### Video Generation
- Uses OpenCV for video processing
- MP4 format with H.264 codec
- Configurable FPS (1-5)
- Frame duration control (0.3-2.0s)

### Image Processing
- PIL and OpenCV integration
- Real-time image analysis
- Mock YOLO detection simulation
- Visual overlay rendering

### Session Management
- Processing history tracking
- Success rate calculation
- Real-time statistics updates
- Session state persistence

## 🎉 Demo Usage

1. **Start the Demo**: Run `run_streamlit_demo.bat` or use manual launch
2. **Navigate Features**: Use sidebar to explore different modules
3. **Try Text-to-Sign**: Enter text and generate sign language videos
4. **Upload Images**: Test sign detection with your images
5. **Learn Alphabet**: Use interactive learning module
6. **View Analytics**: Check your session statistics

## 🔍 Troubleshooting

### Common Issues

**"No sign images found"**
- Ensure sign images (A.png - Z.png) exist in `BackEnd/TextToSignPipeline/sign_images/`
- Check file permissions

**"Module not found" errors**
- Run: `pip install -r streamlit_requirements.txt`
- Ensure Python 3.8+ is installed

**Video generation fails**
- Check available disk space
- Verify OpenCV installation
- Ensure write permissions in temp directory

**Streamlit won't start**
- Check if port 8501 is available
- Try different port: `streamlit run streamlit_demo.py --server.port 8502`
- Restart terminal/command prompt

## 🎮 Demo Tips

- **Best Performance**: Use Chrome or Firefox browsers
- **Video Quality**: Higher FPS = smoother but larger files
- **Learning Mode**: Try random challenges for practice
- **Mobile Support**: Demo works on tablets and phones
- **Offline Use**: No internet required after installation

## 📞 Support

For issues with the demo:
1. Check this README
2. Verify all requirements are installed
3. Ensure sign images are available
4. Check browser console for errors

## 🚀 Next Steps

After trying the demo:
- Explore the full SignBridge backend API
- Integrate with React frontend
- Deploy on cloud platforms
- Add more sign languages
- Enhance detection models

---

**🤟 SignBridge Demo - Bridging Communication Through Technology**