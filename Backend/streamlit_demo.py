"""
SignBridge Demo - Comprehensive Streamlit Application
====================================================

This demo application showcases all features of the SignBridge system:
- Speech-to-Sign Language Conversion
- Text-to-Sign Language Conversion 
- Sign Language Detection
- Interactive Learning Modules
- Real-time Processing

Author: SignBridge Team
Date: September 2025
"""

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
import threading
from pathlib import Path
import json
import base64
from PIL import Image
import io

# Try to import YOLO, gracefully handle if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.sidebar.warning("⚠️ YOLO not available. Install with: pip install ultralytics")

# Page configuration
st.set_page_config(
    page_title="SignBridge Demo",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background-color: #black;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    .demo-section {
        background-color: #black;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
    
    .info-message {
        color: #17a2b8;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

if 'current_video_path' not in st.session_state:
    st.session_state.current_video_path = None

# Sidebar navigation
st.sidebar.title("🤟 SignBridge Demo")
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.selectbox(
    "Choose a Demo Feature",
    [
        "🏠 Home & Overview",
        "🎤 Speech-to-Sign",
        "📝 Text-to-Sign", 
        "👁️ Sign Detection",
        "� Braille Conversion",
        "�📚 Course Materials",
        "🎓 Learning Module",
        "⚡ Real-time Processing",
        "📊 Analytics & Stats"
    ]
)

# Function to load sign images
@st.cache_data
def load_sign_images():
    """Load available sign language images"""
    sign_images_dir = Path("BackEnd/TextToSignPipeline/sign_images")
    available_letters = []
    
    if sign_images_dir.exists():
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            img_path = sign_images_dir / f"{letter}.png"
            if img_path.exists():
                available_letters.append(letter)
    
    return available_letters, sign_images_dir

# Function to create sign video
def create_sign_video_demo(text, fps=2, frame_duration=0.5):
    """Create a sign language video from text"""
    available_letters, sign_images_dir = load_sign_images()
    
    # Clean text - keep only letters
    clean_text = ''.join([char.upper() for char in text if char.upper() in available_letters])
    
    if not clean_text:
        return None, "No valid letters found in the input text"
    
    # Create temporary video file
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"sign_video_{int(time.time())}.mp4")
    
    try:
        # Load first image to get dimensions
        first_img_path = sign_images_dir / f"{clean_text[0]}.png"
        first_img = cv2.imread(str(first_img_path))
        height, width, layers = first_img.shape
        
        # Define video codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_per_letter = int(fps * frame_duration)
        
        # Process each letter
        for letter in clean_text:
            img_path = sign_images_dir / f"{letter}.png"
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.resize(img, (width, height))
                
                # Add text overlay
                cv2.putText(img, letter, (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
                
                # Write frames for this letter
                for _ in range(frames_per_letter):
                    video_writer.write(img)
        
        video_writer.release()
        
        return output_path, f"Video created successfully for: {clean_text}"
        
    except Exception as e:
        return None, f"Error creating video: {str(e)}"

# Function to detect signs in uploaded image
def detect_signs_in_image(image):
    """Detect sign language in uploaded image using YOLO"""
    try:
        # Convert PIL image to CV2 format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Try to load YOLO model
        model = load_yolo_model()
        
        if model is None:
            # Fallback to mock detection if model loading fails
            return mock_sign_detection(opencv_image)
        
        # Apply preprocessing (similar to realtime_inference.py)
        processed_image = apply_sign_preprocessing(opencv_image)
        
        # Run YOLO detection
        results = model(processed_image, conf=0.5)  # confidence threshold
        
        detected_signs = []
        confidence_scores = []
        annotated_image = opencv_image.copy()
        
        # Define class names for ASL signs
        class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Process detection results
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Get class, confidence, and coordinates
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class label
                if cls_id < len(class_names):
                    class_label = class_names[cls_id]
                else:
                    class_label = f"Class_{cls_id}"
                
                detected_signs.append(class_label)
                confidence_scores.append(conf)
                
                # Draw bounding box and label
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Create label with confidence
                label = f"{class_label}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw background for text
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw text
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Convert back to RGB for Streamlit
        result_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return result_image, detected_signs, confidence_scores
        
    except Exception as e:
        st.error(f"Error in YOLO sign detection: {str(e)}")
        return mock_sign_detection(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

def apply_sign_preprocessing(image):
    """Apply preprocessing pipeline optimized for sign language detection"""
    try:
        # Apply bilateral filter for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Apply contrast enhancement using CLAHE
        lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply skin segmentation for hand detection
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply skin mask to enhanced image
        skin_segmented = cv2.bitwise_and(enhanced, enhanced, mask=skin_mask)
        
        return skin_segmented
        
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return image
        
        # Process YOLO results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Map class ID to sign letter (assuming 26 classes for A-Z)
                    if class_id < 26:
                        sign_letter = chr(ord('A') + class_id)
                    else:
                        sign_letter = f"CLASS_{class_id}"
                    
                    detected_signs.append(sign_letter)
                    confidence_scores.append(confidence)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{sign_letter}: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Convert back to RGB for Streamlit
        result_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return result_image, detected_signs, confidence_scores
        
    except Exception as e:
        st.error(f"Error in sign detection: {str(e)}")
        # Fallback to mock detection
        return mock_sign_detection(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

def load_yolo_model():
    """Load YOLO model with caching"""
    try:
        if not YOLO_AVAILABLE:
            st.sidebar.error("❌ YOLO library not available")
            return None
            
        # Check for available models (prioritizing fine_tuned.pt)
        model_paths = [
            "fine_tuned.pt",  # Fine-tuned ASL model in root
            "runs/detect/train2/weights/fine_tuned.pt",  # Alternative location
            "yolov8n.pt",     # YOLO nano in root
            "yolo11n.pt",     # YOLO11 nano in root
            "BackEnd/models/SignConv.pt",  # SignConv model
            "BackEnd/SignDetectionPipeline/models/SignConv.pt",  # SignConv in pipeline
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = YOLO(model_path)
                model_name = os.path.basename(model_path)
                st.sidebar.success(f"✅ YOLO model loaded: {model_name}")
                
                # Show model info
                if model_path == "fine_tuned.pt":
                    st.sidebar.info("🎯 Using fine-tuned ASL model")
                elif "fine_tuned" in model_path:
                    st.sidebar.info("🎯 Using fine-tuned ASL model (alt location)")
                
                return model
        
        # If no local models found, try downloading YOLOv8n
        try:
            model = YOLO('yolov8n.pt')  # This will download if not present
            st.sidebar.warning("📥 Using default YOLOv8n model (not trained for ASL)")
            return model
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load default model: {str(e)}")
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {str(e)}")
        return None

def mock_sign_detection(opencv_image):
    """Fallback mock detection when YOLO model is not available"""
    try:
        height, width = opencv_image.shape[:2]
        
        # Mock detection results
        detected_signs = ["HELLO", "WORLD", "THANK", "YOU"]
        confidence_scores = [0.95, 0.87, 0.92, 0.89]
        
        # Mock bounding box coordinates
        boxes = [
            (50, 50, 200, 200),
            (250, 50, 200, 200),
            (50, 300, 200, 200),
            (250, 300, 200, 200)
        ]
        
        for i, (x, y, w, h) in enumerate(boxes[:len(detected_signs)]):
            if i < len(detected_signs):
                # Draw rectangle
                cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for mock
                
                # Add label
                label = f"{detected_signs[i]}: {confidence_scores[i]:.2f} (DEMO)"
                cv2.putText(opencv_image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Convert back to RGB for Streamlit
        result_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        return result_image, detected_signs, confidence_scores
        
    except Exception as e:
        st.error(f"Error in mock detection: {str(e)}")
        return None, [], []

# Braille conversion function
def text_to_braille_demo(text, contracted=False, enable_haptic=True):
    """Convert text to Braille characters"""
    # Braille character mapping
    braille_map = {
        'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋',
        'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇',
        'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗',
        's': '⠎', 't': '⠞', 'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭',
        'y': '⠽', 'z': '⠵', ' ': '⠀',
        '0': '⠚', '1': '⠁', '2': '⠃', '3': '⠉', '4': '⠙', '5': '⠑',
        '6': '⠋', '7': '⠛', '8': '⠓', '9': '⠊',
        '.': '⠲', ',': '⠂', '?': '⠦', '!': '⠖', ':': '⠒', ';': '⠆'
    }
    
    try:
        clean_text = text.lower().strip()
        braille_text = ""
        character_map = []
        
        for i, char in enumerate(clean_text):
            if char in braille_map:
                braille_char = braille_map[char]
                braille_text += braille_char
                character_map.append({
                    'original': char,
                    'braille': braille_char,
                    'position': i
                })
            else:
                braille_text += '⠿'  # Generic symbol for unknown characters
                character_map.append({
                    'original': char,
                    'braille': '⠿',
                    'position': i
                })
        
        return {
            'success': True,
            'original_text': text,
            'braille': braille_text,
            'character_count': len(clean_text),
            'character_map': character_map
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'original_text': text
        }

# Course materials functions
def get_sample_courses():
    """Get sample course data"""
    return {
        'basic-asl': {
            'id': 'basic-asl',
            'title': 'Basic American Sign Language',
            'description': 'Introduction to fundamental ASL signs and grammar',
            'difficulty': 'Beginner',
            'duration': '4 weeks',
            'lessons': 12,
            'category': 'Sign Language',
            'instructor': 'Dr. Sarah Johnson',
            'rating': 4.8,
            'enrolled_count': 1250,
            'progress': 0
        },
        'intermediate-asl': {
            'id': 'intermediate-asl',
            'title': 'Intermediate ASL Conversation',
            'description': 'Build conversational skills and advanced vocabulary',
            'difficulty': 'Intermediate',
            'duration': '6 weeks',
            'lessons': 18,
            'category': 'Sign Language',
            'instructor': 'Mark Thompson',
            'rating': 4.7,
            'enrolled_count': 890,
            'progress': 0
        },
        'deaf-culture': {
            'id': 'deaf-culture',
            'title': 'Understanding Deaf Culture',
            'description': 'Learn about Deaf community history, values, and traditions',
            'difficulty': 'Beginner',
            'duration': '3 weeks',
            'lessons': 8,
            'category': 'Cultural',
            'instructor': 'Lisa Williams',
            'rating': 4.9,
            'enrolled_count': 670,
            'progress': 0
        },
        'fingerspelling-mastery': {
            'id': 'fingerspelling-mastery',
            'title': 'Fingerspelling Mastery',
            'description': 'Master the ASL alphabet and number system',
            'difficulty': 'Beginner',
            'duration': '2 weeks',
            'lessons': 6,
            'category': 'Fundamentals',
            'instructor': 'Jennifer Davis',
            'rating': 4.6,
            'enrolled_count': 2100,
            'progress': 0
        }
    }

# Main application logic based on selected page
if page == "🏠 Home & Overview":
    st.markdown('<h1 class="main-header">🤟 Welcome to SignBridge Demo</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-section">
    <h2>🌟 About SignBridge</h2>
    <p>SignBridge is a comprehensive AI-powered platform that bridges communication gaps between hearing and deaf communities through advanced sign language processing technologies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>🎤 Speech-to-Sign</h3>
        <p>Convert spoken words into sign language videos in real-time</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h3>🔤 Braille Conversion</h3>
        <p>Transform text into Braille characters with haptic feedback</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>📝 Text-to-Sign</h3>
        <p>Transform written text into animated sign language sequences</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h3>📚 Course Materials</h3>
        <p>Access educational content and track learning progress</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h3>👁️ Sign Detection</h3>
        <p>Recognize and interpret sign language from images and videos</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h3>🎓 Interactive Learning</h3>
        <p>Practice and master sign language through guided modules</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("### 📊 System Statistics")
    
    available_letters, _ = load_sign_images()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Available Signs", len(available_letters))
    
    with col2:
        st.metric("Braille Characters", "63+")
    
    with col3:
        st.metric("Course Modules", "5+")
    
    with col4:
        st.metric("Learning Features", "8")

elif page == "🎤 Speech-to-Sign":
    st.markdown('<h1 class="main-header">🎤 Speech-to-Sign Conversion</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-section">
    <h2>Convert Speech to Sign Language</h2>
    <p>This demo simulates speech-to-sign conversion. In a real implementation, this would use speech recognition APIs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulated speech input
    st.subheader("Simulate Speech Input")
    
    speech_options = [
        "HELLO WORLD",
        "THANK YOU",
        "HOW ARE YOU",
        "NICE TO MEET YOU",
        "GOODBYE"
    ]
    
    selected_speech = st.selectbox("Select speech to convert:", speech_options)
    custom_speech = st.text_input("Or enter custom text:", placeholder="Type any text here...")
    
    speech_text = custom_speech.upper() if custom_speech else selected_speech
    
    # Audio settings
    st.subheader("Video Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        fps = st.slider("Video FPS", min_value=1, max_value=5, value=2)
    
    with col2:
        frame_duration = st.slider("Frame Duration (seconds)", min_value=0.3, max_value=2.0, value=0.5, step=0.1)
    
    # Convert button
    if st.button("🎤 Convert Speech to Sign Language", type="primary"):
        if speech_text:
            with st.spinner("Converting speech to sign language..."):
                video_path, message = create_sign_video_demo(speech_text, fps, frame_duration)
                
                if video_path and os.path.exists(video_path):
                    st.success(message)
                    st.session_state.current_video_path = video_path
                    
                    # Display video
                    st.subheader("Generated Sign Language Video")
                    st.video(video_path)
                    
                    # Download button
                    with open(video_path, 'rb') as file:
                        st.download_button(
                            label="📥 Download Video",
                            data=file.read(),
                            file_name=f"speech_to_sign_{int(time.time())}.mp4",
                            mime="video/mp4"
                        )
                    
                    # Add to history
                    st.session_state.processing_history.append({
                        'type': 'Speech-to-Sign',
                        'input': speech_text,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'success': True
                    })
                else:
                    st.error(message)
        else:
            st.warning("Please enter some text to convert.")

elif page == "📝 Text-to-Sign":
    st.markdown('<h1 class="main-header">📝 Text-to-Sign Conversion</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-section">
    <h2>Convert Text to Sign Language Videos</h2>
    <p>Enter any text and watch it transform into animated sign language sequences.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Text input
    st.subheader("Enter Text to Convert")
    
    text_input = st.text_area(
        "Text Input:",
        height=100,
        placeholder="Enter the text you want to convert to sign language..."
    )
    
    # Preview letters
    if text_input:
        available_letters, _ = load_sign_images()
        clean_text = ''.join([char.upper() for char in text_input if char.upper() in available_letters])
        
        if clean_text:
            st.info(f"Letters to be converted: {' '.join(list(clean_text))}")
        else:
            st.warning("No valid letters found in the input text.")
    
    # Video settings
    st.subheader("Video Generation Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fps = st.slider("Video FPS", min_value=1, max_value=5, value=2, key="text_fps")
    
    with col2:
        frame_duration = st.slider("Frame Duration (seconds)", min_value=0.3, max_value=2.0, value=0.5, step=0.1, key="text_duration")
    
    with col3:
        show_text = st.checkbox("Show text overlay", value=True)
    
    # Generate video button
    if st.button("🎬 Generate Sign Language Video", type="primary"):
        if text_input.strip():
            with st.spinner("Generating sign language video..."):
                video_path, message = create_sign_video_demo(text_input, fps, frame_duration)
                
                if video_path and os.path.exists(video_path):
                    st.success(message)
                    
                    # Display video
                    st.subheader("Generated Sign Language Video")
                    st.video(video_path)
                    
                    # Download button
                    with open(video_path, 'rb') as file:
                        st.download_button(
                            label="📥 Download Video",
                            data=file.read(),
                            file_name=f"text_to_sign_{int(time.time())}.mp4",
                            mime="video/mp4"
                        )
                    
                    # Add to history
                    st.session_state.processing_history.append({
                        'type': 'Text-to-Sign',
                        'input': text_input,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'success': True
                    })
                else:
                    st.error(message)
        else:
            st.warning("Please enter some text to convert.")

elif page == "👁️ Sign Detection":
    st.markdown('<h1 class="main-header">👁️ Advanced Sign Language Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-section">
    <h2>AI-Powered Sign Language Recognition</h2>
    <p>Upload images or use your camera to detect and interpret sign language gestures with advanced YOLO models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model status indicator
    model = load_yolo_model()
    if model is not None:
        st.success("🤖 AI Model Status: Ready for detection!")
    else:
        st.warning("⚠️ AI Model not available - using simulation mode")
    
    # Detection method tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Camera Detection", "⚙️ Advanced Settings"])
    
    with tab1:
        st.subheader("Upload Image for Detection")
        
        # File uploader with more options
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing sign language gestures"
        )
        
        # Detection settings
        col_set1, col_set2, col_set3 = st.columns(3)
        
        with col_set1:
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                help="Minimum confidence score for detections"
            )
        
        with col_set2:
            max_detections = st.slider(
                "Max Detections", 
                min_value=1, max_value=20, value=10,
                help="Maximum number of signs to detect"
            )
        
        with col_set3:
            apply_preprocessing = st.checkbox(
                "Apply Preprocessing", 
                value=True,
                help="Use hand segmentation and image enhancement"
            )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                width, height = image.size
                st.info(f"📊 Image: {width}x{height} pixels, {image.mode} mode")
            
            # Detect signs button
            if st.button("🔍 Analyze Sign Language", type="primary", key="detect_upload"):
                with st.spinner("🤖 AI is analyzing the image for sign language..."):
                    start_time = time.time()
                    
                    result_image, detected_signs, confidence_scores = detect_signs_in_image(image)
                    
                    processing_time = time.time() - start_time
                    
                    with col2:
                        st.subheader("🎯 Detection Results")
                        st.image(result_image, caption="Detection Results", use_column_width=True)
                        
                        st.info(f"⚡ Processing time: {processing_time:.2f}s")
                    
                    # Display detailed results
                    if detected_signs:
                        st.success(f"✅ Found {len(detected_signs)} sign(s) in the image!")
                        
                        # Results table
                        st.subheader("📋 Detected Signs")
                        results_data = []
                        
                        for i, (sign, confidence) in enumerate(zip(detected_signs, confidence_scores)):
                            results_data.append({
                                'Rank': i + 1,
                                'Sign': sign,
                                'Confidence': f"{confidence:.1%}",
                                'Category': 'Letter' if sign.isalpha() else 'Number' if sign.isdigit() else 'Symbol'
                            })
                        
                        st.table(results_data)
                        
                        # Statistics
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        with col_stat1:
                            st.metric("Total Signs", len(detected_signs))
                        
                        with col_stat2:
                            avg_confidence = sum(confidence_scores) / len(confidence_scores)
                            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                        
                        with col_stat3:
                            max_confidence = max(confidence_scores)
                            st.metric("Best Detection", f"{max_confidence:.1%}")
                        
                        with col_stat4:
                            letters_count = sum(1 for sign in detected_signs if sign.isalpha())
                            st.metric("Letters Found", letters_count)
                        
                        # Interpretation section
                        if len(detected_signs) > 1:
                            st.subheader("🔤 Possible Word/Phrase")
                            interpreted_text = ''.join(detected_signs[:10])  # Limit to first 10
                            st.markdown(f"**Detected sequence:** `{interpreted_text}`")
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            'type': 'Sign Detection',
                            'input': f"Image with {len(detected_signs)} detected signs: {', '.join(detected_signs[:3])}" + ('...' if len(detected_signs) > 3 else ''),
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'success': True
                        })
                        
                    else:
                        st.warning("⚠️ No signs detected in the image. Try adjusting the confidence threshold or use a clearer image.")
                        st.info("💡 Tips for better detection:\n- Ensure good lighting\n- Clear hand gestures\n- Minimal background distractions")
    
    with tab2:
        st.subheader("Real-time Camera Detection")
        st.info("🚧 Camera detection feature is under development")
        
        # Placeholder for camera functionality
        if st.button("📷 Start Camera Detection", key="start_camera"):
            st.info("📹 This would normally start your camera and detect signs in real-time")
            st.info("🔧 Implementation requires streamlit-webrtc or similar for browser camera access")
        
        # Simulation of real-time detection
        st.subheader("📺 Simulated Real-time Detection")
        
        if st.button("▶️ Start Simulation", key="sim_realtime"):
            # Create a placeholder for updating results
            status_placeholder = st.empty()
            result_placeholder = st.empty()
            
            # Simulate real-time detection
            sample_detections = [
                ("A", 0.89, time.time()),
                ("B", 0.76, time.time() + 1),
                ("C", 0.92, time.time() + 2),
                ("HELLO", 0.85, time.time() + 3),
            ]
            
            for sign, conf, timestamp in sample_detections:
                status_placeholder.info(f"📹 Detecting: {sign} (confidence: {conf:.1%})")
                result_placeholder.success(f"✅ Latest detection: **{sign}** at {time.strftime('%H:%M:%S', time.localtime(timestamp))}")
                time.sleep(1)
            
            status_placeholder.success("✅ Simulation complete!")
    
    with tab3:
        st.subheader("⚙️ Advanced Detection Settings")
        
        # Model settings
        st.markdown("#### 🤖 Model Configuration")
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            iou_threshold = st.slider(
                "IoU Threshold", 
                min_value=0.1, max_value=1.0, value=0.45, step=0.05,
                help="Intersection over Union threshold for non-maximum suppression"
            )
            
            class_agnostic_nms = st.checkbox(
                "Class Agnostic NMS", 
                value=False,
                help="Apply non-maximum suppression across all classes"
            )
        
        with col_adv2:
            max_det = st.slider(
                "Maximum Detections", 
                min_value=10, max_value=1000, value=300,
                help="Maximum number of detections per image"
            )
            
            augment = st.checkbox(
                "Test Time Augmentation", 
                value=False,
                help="Apply test-time augmentation for better accuracy"
            )
        
        # Preprocessing settings
        st.markdown("#### 🖼️ Image Preprocessing")
        
        col_prep1, col_prep2 = st.columns(2)
        
        with col_prep1:
            enable_skin_seg = st.checkbox(
                "Enable Skin Segmentation", 
                value=True,
                help="Focus detection on skin-colored regions"
            )
            
            contrast_enhancement = st.checkbox(
                "Contrast Enhancement", 
                value=True,
                help="Apply CLAHE for better contrast"
            )
        
        with col_prep2:
            bilateral_filter = st.checkbox(
                "Noise Reduction", 
                value=True,
                help="Apply bilateral filtering for noise reduction"
            )
            
            resize_input = st.checkbox(
                "Resize Input", 
                value=True,
                help="Resize images for consistent processing"
            )
        
        # Display current settings
        st.markdown("#### 📊 Current Settings Summary")
        
        settings_summary = {
            "Model Available": "Yes" if model else "No",
            "Confidence Threshold": f"{confidence_threshold:.2f}",
            "IoU Threshold": f"{iou_threshold:.2f}",
            "Max Detections": max_detections,
            "Preprocessing": "Enabled" if apply_preprocessing else "Disabled"
        }
        
        for setting, value in settings_summary.items():
            st.write(f"**{setting}:** {value}")
        
        if st.button("🔄 Reset to Defaults", key="reset_settings"):
            st.experimental_rerun()

    # Main detection tab
    with tab1:
        st.write("Upload an image to detect ASL signs:")
        
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'], key="main_upload")
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process the image
            start_time = time.time()
            result_image, detected_signs, confidence_scores = detect_signs_in_image(image)
            processing_time = time.time() - start_time
            
            if result_image is not None:
                with col2:
                    st.subheader("🎯 Detection Results")
                    st.image(result_image, caption="Detected Signs", use_column_width=True)
                
                # Processing metrics
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric("🔍 Signs Found", len(detected_signs))
                
                with col_metric2:
                    avg_conf = np.mean(confidence_scores) if confidence_scores else 0
                    st.metric("🎯 Avg Confidence", f"{avg_conf:.1%}")
                
                with col_metric3:
                    st.metric("⚡ Processing Time", f"{processing_time:.2f}s")
                
                # Detailed results
                if detected_signs:
                    st.subheader("📋 Detected Signs Details")
                    
                    # Create detailed results table
                    results_data = []
                    for i, (sign, confidence) in enumerate(zip(detected_signs, confidence_scores)):
                        results_data.append({
                            '#': i + 1,
                            'Sign': sign,
                            'Confidence': f"{confidence:.2%}",
                            'Quality': "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
                        })
                    
                    st.table(results_data)
                    
                    # Generate transcription
                    transcription = " ".join(detected_signs)
                    st.subheader("📝 Transcription")
                    st.markdown(f"**Detected Message:** `{transcription}`")
                    
                    # Download results
                    results_text = f"""Sign Language Detection Results
                    
Image: {uploaded_file.name}
Processing Time: {processing_time:.2f} seconds
Confidence Threshold: {confidence_threshold}

Detected Signs:
{chr(10).join([f"- {sign}: {conf:.2%}" for sign, conf in zip(detected_signs, confidence_scores)])}

Transcription: {transcription}

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
                    
                    st.download_button(
                        label="📥 Download Results",
                        data=results_text,
                        file_name=f"sign_detection_{int(time.time())}.txt",
                        mime="text/plain"
                    )
                    
                    # Add to history
                    st.session_state.processing_history.append({
                        'type': 'Sign Detection',
                        'input': f"Image: {len(detected_signs)} signs detected",
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'success': True
                    })
                    
                else:
                    st.warning("🚫 No signs detected in the image. Try adjusting the confidence threshold or uploading a clearer image.")
            else:
                st.error("❌ Error processing the image. Please try again.")
    
    with tab2:
        st.subheader("📷 Real-time Camera Detection")
        st.info("🚧 Camera integration is coming soon! This feature will enable real-time sign language detection using your device's camera.")
        
        # Placeholder for camera functionality
        if st.button("🎥 Start Camera Detection", disabled=True):
            st.info("Camera detection will be available in future updates.")
    
    with tab3:
        st.subheader("⚙️ Detection Settings & Model Info")
        
        # Model information
        st.markdown("### 🤖 AI Model Information")
        
        model_info_col1, model_info_col2 = st.columns(2)
        
        with model_info_col1:
            # Check available models
            model_paths = [
                ("fine_tuned.pt", "Fine-tuned SignBridge Model"),
                ("yolov8n.pt", "YOLOv8 Nano"),
                ("BackEnd/models/SignConv.pt", "SignConv Model"),
                ("BackEnd/SignDetectionPipeline/models/SignConv.pt", "Pipeline SignConv")
            ]
            
            st.markdown("**Available Models:**")
            for model_path, model_name in model_paths:
                if os.path.exists(model_path):
                    st.success(f"✅ {model_name}")
                else:
                    st.error(f"❌ {model_name}")
        
        with model_info_col2:
            st.markdown("**Detection Capabilities:**")
            st.write("• A-Z American Sign Language letters")
            st.write("• Real-time processing")
            st.write("• Confidence scoring")
            st.write("• Multiple detection support")
            st.write("• Bounding box visualization")
        
        # Performance settings
        st.markdown("### 🎛️ Performance Settings")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.markdown("**Recommended Settings:**")
            st.write("• Confidence: 0.5-0.7 for general use")
            st.write("• Max Detections: 5-10 for clarity")
            st.write("• Image resolution: 640x640 optimal")
        
        with perf_col2:
            st.markdown("**Tips for Better Detection:**")
            st.write("• Use good lighting conditions")
            st.write("• Clear hand positioning")
            st.write("• Minimal background noise")
            st.write("• Steady hand gestures")
        
        # System status
        st.markdown("### 📊 System Status")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            st.metric("Model Status", "Ready" if load_yolo_model() else "Error")
        
        with status_col2:
            st.metric("Detection Speed", "Real-time")
        
        with status_col3:
            st.metric("Supported Formats", "5 types")
        
        # Test detection button
        if st.button("🧪 Test Model Loading"):
            with st.spinner("Testing model..."):
                model = load_yolo_model()
                if model:
                    st.success("✅ Model loaded successfully!")
                    st.balloons()
                else:
                    st.error("❌ Model loading failed. Check the logs above.")

elif page == "� Braille Conversion":
    st.markdown('<h1 class="main-header">🔤 Braille Conversion System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-section">
    <h2>Convert Text to Braille</h2>
    <p>Transform any text into Braille characters with haptic feedback and multi-language support.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Text input for Braille conversion
    st.subheader("Text to Braille Conversion")
    
    braille_text = st.text_area(
        "Enter text to convert to Braille:",
        height=100,
        placeholder="Type any text here to convert to Braille...",
        key="braille_input"
    )
    
    # Conversion settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        contracted_braille = st.checkbox("Use Contracted Braille", help="Use shortened forms for common words")
    
    with col2:
        enable_haptic = st.checkbox("Enable Haptic Feedback", value=True, help="Simulate vibration patterns")
    
    with col3:
        audio_feedback = st.checkbox("Audio Feedback", help="Text-to-speech for accessibility")
    
    # Sample text suggestions
    st.subheader("Quick Text Samples")
    sample_texts = [
        "Hello World",
        "How are you today?",
        "Thank you very much",
        "Good morning",
        "Sign language is amazing!"
    ]
    
    sample_cols = st.columns(len(sample_texts))
    for i, sample in enumerate(sample_texts):
        with sample_cols[i]:
            if st.button(sample, key=f"sample_{i}"):
                st.session_state.braille_input = sample
                st.experimental_rerun()
    
    # Convert to Braille
    if st.button("🔤 Convert to Braille", type="primary"):
        if braille_text.strip():
            with st.spinner("Converting to Braille..."):
                result = text_to_braille_demo(braille_text, contracted_braille, enable_haptic)
                
                if result['success']:
                    st.success("Text converted to Braille successfully!")
                    
                    # Display results
                    st.subheader("Braille Output")
                    
                    # Main Braille text display
                    st.markdown(f"""
                    <div style="
                        background-color: #f8f9fa;
                        padding: 2rem;
                        border-radius: 10px;
                        border: 2px solid #dee2e6;
                        margin: 1rem 0;
                        text-align: center;
                    ">
                        <h3>Original Text:</h3>
                        <p style="font-size: 1.2rem; margin: 1rem 0;">{result['original_text']}</p>
                        <h3>Braille:</h3>
                        <p style="font-size: 2rem; font-family: monospace; line-height: 1.5; color: #0066cc;">
                            {result['braille']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Statistics and details
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Characters", result['character_count'])
                    
                    with col2:
                        st.metric("Braille Characters", len(result['braille'].replace(' ', '')))
                    
                    with col3:
                        st.metric("Conversion Rate", "100%")
                    
                    # Character mapping table
                    if st.expander("View Character Mapping", expanded=False):
                        char_data = []
                        for mapping in result['character_map'][:20]:  # Show first 20 characters
                            char_data.append({
                                'Original': mapping['original'] if mapping['original'] != ' ' else '(space)',
                                'Braille': mapping['braille'],
                                'Position': mapping['position'] + 1
                            })
                        
                        if char_data:
                            st.table(char_data)
                        
                        if len(result['character_map']) > 20:
                            st.info(f"Showing first 20 characters. Total: {len(result['character_map'])}")
                    
                    # Download options
                    st.subheader("Download Options")
                    
                    download_text = f"""Original Text:
{result['original_text']}

Braille Conversion:
{result['braille']}

Conversion Details:
- Character Count: {result['character_count']}
- Contracted Braille: {'Yes' if contracted_braille else 'No'}
- Haptic Enabled: {'Yes' if enable_haptic else 'No'}
- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
                    
                    st.download_button(
                        label="📥 Download Braille Text",
                        data=download_text,
                        file_name=f"braille_conversion_{int(time.time())}.txt",
                        mime="text/plain"
                    )
                    
                    # Audio feedback simulation
                    if audio_feedback:
                        if st.button("🔊 Play Audio (Simulated)", key="play_audio"):
                            st.info("🔊 Audio playback simulation: Reading text aloud...")
                            st.balloons()
                    
                    # Haptic feedback simulation
                    if enable_haptic:
                        if st.button("📳 Test Haptic Pattern", key="test_haptic"):
                            st.info("📳 Haptic feedback simulation: Vibration pattern activated!")
                            st.success("Each character has a unique vibration pattern for tactile learning.")
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        'type': 'Braille Conversion',
                        'input': braille_text[:50] + ('...' if len(braille_text) > 50 else ''),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'success': True
                    })
                    
                else:
                    st.error(f"Conversion failed: {result.get('error', 'Unknown error')}")
        else:
            st.warning("Please enter some text to convert to Braille.")
    
    # Braille reference guide
    st.subheader("🔍 Braille Reference Guide")
    
    with st.expander("View Braille Alphabet", expanded=False):
        braille_ref = {
            'A': '⠁', 'B': '⠃', 'C': '⠉', 'D': '⠙', 'E': '⠑', 'F': '⠋',
            'G': '⠛', 'H': '⠓', 'I': '⠊', 'J': '⠚', 'K': '⠅', 'L': '⠇',
            'M': '⠍', 'N': '⠝', 'O': '⠕', 'P': '⠏', 'Q': '⠟', 'R': '⠗',
            'S': '⠎', 'T': '⠞', 'U': '⠥', 'V': '⠧', 'W': '⠺', 'X': '⠭',
            'Y': '⠽', 'Z': '⠵'
        }
        
        # Display alphabet in grid
        cols = st.columns(6)
        for i, (letter, braille) in enumerate(braille_ref.items()):
            with cols[i % 6]:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 5px;">
                    <strong>{letter}</strong><br>
                    <span style="font-size: 1.5rem;">{braille}</span>
                </div>
                """, unsafe_allow_html=True)

elif page == "📚 Course Materials":
    st.markdown('<h1 class="main-header">�📚 Course Materials Platform</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-section">
    <h2>Educational Content Management</h2>
    <p>Browse and manage sign language learning courses and educational materials.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get sample courses
    courses = get_sample_courses()
    
    # Course filters
    st.subheader("📋 Course Catalog")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        difficulty_filter = st.selectbox(
            "Filter by Difficulty:",
            ["All Levels", "Beginner", "Intermediate", "Advanced"]
        )
    
    with col2:
        category_filter = st.selectbox(
            "Filter by Category:",
            ["All Categories", "Sign Language", "Cultural", "Fundamentals", "Professional"]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ["Rating", "Popularity", "Duration", "Title"]
        )
    
    # Search bar
    search_query = st.text_input("🔍 Search courses:", placeholder="Enter course title or instructor name...")
    
    # Filter courses
    filtered_courses = list(courses.values())
    
    if difficulty_filter != "All Levels":
        filtered_courses = [c for c in filtered_courses if c['difficulty'] == difficulty_filter]
    
    if category_filter != "All Categories":
        filtered_courses = [c for c in filtered_courses if c['category'] == category_filter]
    
    if search_query:
        search_lower = search_query.lower()
        filtered_courses = [c for c in filtered_courses if 
                          search_lower in c['title'].lower() or 
                          search_lower in c['instructor'].lower()]
    
    # Sort courses
    if sort_by == "Rating":
        filtered_courses.sort(key=lambda x: x['rating'], reverse=True)
    elif sort_by == "Popularity":
        filtered_courses.sort(key=lambda x: x['enrolled_count'], reverse=True)
    elif sort_by == "Duration":
        filtered_courses.sort(key=lambda x: x['lessons'])
    else:  # Title
        filtered_courses.sort(key=lambda x: x['title'])
    
    # Display courses
    if filtered_courses:
        st.info(f"Found {len(filtered_courses)} courses")
        
        for course in filtered_courses:
            with st.expander(f"📖 {course['title']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {course['description']}")
                    st.write(f"**Instructor:** {course['instructor']}")
                    st.write(f"**Duration:** {course['duration']} • {course['lessons']} lessons")
                    st.write(f"**Category:** {course['category']} • **Level:** {course['difficulty']}")
                
                with col2:
                    st.metric("Rating", f"{course['rating']}/5.0", "⭐")
                    st.metric("Enrolled", f"{course['enrolled_count']:,}")
                    st.metric("Progress", f"{course['progress']:.0f}%")
                
                # Action buttons
                button_col1, button_col2, button_col3 = st.columns(3)
                
                with button_col1:
                    if st.button(f"📚 Enroll", key=f"enroll_{course['id']}"):
                        st.success(f"✅ Enrolled in '{course['title']}'!")
                        course['progress'] = 5  # Simulate some progress
                        
                        # Add to history
                        st.session_state.processing_history.append({
                            'type': 'Course Enrollment',
                            'input': course['title'],
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'success': True
                        })
                
                with button_col2:
                    if st.button(f"▶️ Preview", key=f"preview_{course['id']}"):
                        st.info(f"🎥 Opening preview for '{course['title']}'...")
                
                with button_col3:
                    if st.button(f"⭐ Rate", key=f"rate_{course['id']}"):
                        st.info("⭐ Rating feature coming soon!")
    
    else:
        st.warning("No courses found matching your criteria.")
    
    # Course statistics
    st.subheader("📊 Platform Statistics")
    
    total_courses = len(courses)
    total_students = sum(course['enrolled_count'] for course in courses.values())
    avg_rating = sum(course['rating'] for course in courses.values()) / total_courses if total_courses > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Courses", total_courses)
    
    with col2:
        st.metric("Total Students", f"{total_students:,}")
    
    with col3:
        st.metric("Average Rating", f"{avg_rating:.1f}/5.0")
    
    with col4:
        st.metric("Categories", "4")
    
    # My Courses section
    st.subheader("📝 My Enrolled Courses")
    
    enrolled_courses = [course for course in courses.values() if course['progress'] > 0]
    
    if enrolled_courses:
        for course in enrolled_courses:
            st.markdown(f"""
            <div class="feature-card">
                <h4>📚 {course['title']}</h4>
                <p><strong>Progress:</strong> {course['progress']:.0f}% complete</p>
                <p><strong>Instructor:</strong> {course['instructor']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No enrolled courses yet. Browse the catalog above to get started!")

elif page == "🎓 Learning Module":
    st.markdown('<h1 class="main-header">📚 Interactive Learning Module</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-section">
    <h2>Learn Sign Language Alphabet</h2>
    <p>Interactive module to learn and practice sign language letters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    available_letters, sign_images_dir = load_sign_images()
    
    if available_letters:
        # Letter selector
        st.subheader("Choose a Letter to Learn")
        
        selected_letter = st.selectbox("Select Letter:", available_letters)
        
        if selected_letter:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display sign image
                img_path = sign_images_dir / f"{selected_letter}.png"
                if img_path.exists():
                    st.image(str(img_path), caption=f"Sign for letter '{selected_letter}'", width=300)
            
            with col2:
                st.subheader(f"Letter: {selected_letter}")
                st.markdown("### Practice Tips:")
                st.markdown(f"- Focus on the hand position for '{selected_letter}'")
                st.markdown("- Practice the gesture slowly at first")
                st.markdown("- Ensure clear finger positioning")
                st.markdown("- Practice in front of a mirror")
        
        # Practice mode
        st.subheader("Practice Mode")
        
        if st.button("🎯 Random Letter Challenge"):
            random_letter = np.random.choice(available_letters)
            st.info(f"Practice signing the letter: **{random_letter}**")
            
            img_path = sign_images_dir / f"{random_letter}.png"
            if img_path.exists():
                st.image(str(img_path), width=200)
        
        # Statistics
        st.subheader("Learning Progress")
        st.info(f"Available letters to learn: {len(available_letters)}/26")
        
        if len(available_letters) == 26:
            st.success("🎉 Complete alphabet available for learning!")
        else:
            missing_letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - set(available_letters)
            st.warning(f"Missing letters: {', '.join(sorted(missing_letters))}")
    
    else:
        st.error("No sign language images found. Please check the sign_images directory.")

elif page == "⚡ Real-time Processing":
    st.markdown('<h1 class="main-header">⚡ Real-time Processing Demo</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-section">
    <h2>Real-time Text Processing</h2>
    <p>Type text and see instant sign language conversion preview.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time text input
    realtime_text = st.text_input(
        "Type here for real-time preview:",
        key="realtime_input",
        placeholder="Start typing to see live preview..."
    )
    
    if realtime_text:
        available_letters, sign_images_dir = load_sign_images()
        clean_text = ''.join([char.upper() for char in realtime_text if char.upper() in available_letters])
        
        if clean_text:
            st.subheader("Live Preview")
            
            # Display sign images for each letter
            cols = st.columns(min(len(clean_text), 8))  # Max 8 columns
            
            for i, letter in enumerate(clean_text[:8]):  # Show first 8 letters
                with cols[i % len(cols)]:
                    img_path = sign_images_dir / f"{letter}.png"
                    if img_path.exists():
                        st.image(str(img_path), caption=letter, width=100)
            
            if len(clean_text) > 8:
                st.info(f"Showing first 8 letters. Total letters: {len(clean_text)}")
            
            # Generate full video button
            if st.button("🎬 Generate Full Video", key="realtime_video"):
                with st.spinner("Generating video..."):
                    video_path, message = create_sign_video_demo(clean_text)
                    
                    if video_path:
                        st.success(message)
                        st.video(video_path)

elif page == "📊 Analytics & Stats":
    st.markdown('<h1 class="main-header">📊 Analytics & Statistics</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-section">
    <h2>Processing History & Analytics</h2>
    <p>View your session history and system performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System information
    available_letters, sign_images_dir = load_sign_images()
    
    st.subheader("System Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Available Signs", len(available_letters))
    
    with col2:
        st.metric("Total Processes", len(st.session_state.processing_history))
    
    with col3:
        successful_processes = len([h for h in st.session_state.processing_history if h.get('success')])
        st.metric("Successful Processes", successful_processes)
    
    with col4:
        if st.session_state.processing_history:
            success_rate = successful_processes / len(st.session_state.processing_history) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "N/A")
    
    # Processing history
    st.subheader("Processing History")
    
    if st.session_state.processing_history:
        for i, entry in enumerate(reversed(st.session_state.processing_history)):
            with st.expander(f"{entry['type']} - {entry['timestamp']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Input:** {entry['input']}")
                    st.write(f"**Type:** {entry['type']}")
                    st.write(f"**Time:** {entry['timestamp']}")
                with col2:
                    if entry['success']:
                        st.success("✅ Success")
                    else:
                        st.error("❌ Failed")
    else:
        st.info("No processing history yet. Try using other features to see analytics.")
    
    # Clear history button
    if st.session_state.processing_history:
        if st.button("🗑️ Clear History"):
            st.session_state.processing_history = []
            st.success("History cleared!")
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>🤟 SignBridge Demo Application - Bridging Communication Through Technology</p>
    <p>Built with Streamlit • Powered by AI • Made with ❤️</p>
    </div>
    """,
    unsafe_allow_html=True
)