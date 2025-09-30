"""
Text-to-Sign Blueprint
Handles text-to-sign language conversion with video generation and quiz features
"""

from flask import Blueprint, request, jsonify, send_file, Response, stream_with_context
import os
import re
import cv2
import numpy as np
from PIL import Image
import tempfile
import logging
from typing import List, Optional, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import threading
import time
import json
import random
import base64
import io
from datetime import datetime, timedelta

# Create blueprint
text_to_sign_bp = Blueprint('text_to_sign', __name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Processing status tracker
class ProcessingStatus:
    def __init__(self):
        self.status = {}
        self.lock = threading.Lock()
    
    def start_new_task(self, task_id):
        with self.lock:
            self.status[task_id] = {
                'progress': 0,
                'status': 'processing',
                'message': 'Initializing...',
                'start_time': datetime.now().isoformat(),
                'completed': False,
                'result': None
            }
        return task_id
    
    def update_progress(self, task_id, progress, message=None):
        with self.lock:
            if task_id in self.status:
                self.status[task_id]['progress'] = progress
                if message:
                    self.status[task_id]['message'] = message
    
    def complete_task(self, task_id, result=None):
        with self.lock:
            if task_id in self.status:
                self.status[task_id]['progress'] = 100
                self.status[task_id]['status'] = 'completed'
                self.status[task_id]['message'] = 'Task completed'
                self.status[task_id]['completed'] = True
                self.status[task_id]['result'] = result
                self.status[task_id]['end_time'] = datetime.now().isoformat()
    
    def fail_task(self, task_id, error_message):
        with self.lock:
            if task_id in self.status:
                self.status[task_id]['status'] = 'failed'
                self.status[task_id]['message'] = f'Error: {error_message}'
                self.status[task_id]['completed'] = True
                self.status[task_id]['end_time'] = datetime.now().isoformat()
    
    def get_status(self, task_id):
        with self.lock:
            return self.status.get(task_id, {'status': 'not_found'})
    
    def clean_old_tasks(self, max_age_hours=24):
        with self.lock:
            now = datetime.now()
            for task_id in list(self.status.keys()):
                start_time = datetime.fromisoformat(self.status[task_id]['start_time'])
                if (now - start_time).total_seconds() / 3600 > max_age_hours:
                    del self.status[task_id]

# Initialize status tracker
processing_status = ProcessingStatus()

# In-memory storage for quiz sessions (use Redis in production)
quiz_sessions = {}
quiz_leaderboard = []

class TextSummarizer:
    """Simple extractive text summarizer"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logging.warning("NLTK stopwords not available, using basic set")
            self.stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with'}
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return text
            
            word_freq = self._calculate_word_frequency(text)
            sentence_scores = self._score_sentences(sentences, word_freq)
            
            top_sentences = sorted(sentence_scores.items(), 
                                 key=lambda x: x[1], reverse=True)[:max_sentences]
            
            selected_sentences = sorted([sent for sent, score in top_sentences], 
                                      key=lambda x: sentences.index(x))
            
            return ' '.join(selected_sentences)
            
        except Exception as e:
            logging.error(f"Summarization error: {e}")
            sentences = sent_tokenize(text) if text else ['']
            return ' '.join(sentences[:max_sentences])
    
    def _calculate_word_frequency(self, text: str) -> Dict[str, int]:
        try:
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            return Counter(words)
        except:
            words = text.lower().split()
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            return Counter(words)
    
    def _score_sentences(self, sentences: List[str], word_freq: Dict[str, int]) -> Dict[str, float]:
        sentence_scores = {}
        for sentence in sentences:
            try:
                words = word_tokenize(sentence.lower())
            except:
                words = sentence.lower().split()
            
            words = [word for word in words if word.isalnum()]
            if len(words) > 0:
                score = sum(word_freq.get(word, 0) for word in words) / len(words)
                sentence_scores[sentence] = score
            else:
                sentence_scores[sentence] = 0.0
        return sentence_scores

# Initialize text summarizer
text_summarizer = TextSummarizer()

def get_sign_images_directory(app_context=None):
    """Get the path to sign images directory"""
    if app_context:
        return app_context.config.get('TEXT_TO_SIGN_IMAGES_DIR', 'TextToSignPipeline/sign_images')
    else:
        from flask import current_app
        return current_app.config.get('TEXT_TO_SIGN_IMAGES_DIR', 'TextToSignPipeline/sign_images')

def get_output_directory(app_context=None):
    """Get the path to output directory"""
    if app_context:
        return app_context.config.get('TEXT_TO_SIGN_OUTPUT_DIR', 'TextToSignPipeline/output_videos')
    else:
        from flask import current_app
        return current_app.config.get('TEXT_TO_SIGN_OUTPUT_DIR', 'TextToSignPipeline/output_videos')

def clean_text_for_signs(text: str) -> str:
    """Clean and prepare text for sign conversion"""
    # Convert to uppercase and remove special characters
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.upper().strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text

def get_available_letters(app_context=None) -> List[str]:
    """Get list of available sign letters"""
    images_dir = get_sign_images_directory(app_context)
    available_letters = []
    
    if os.path.exists(images_dir):
        for file in os.listdir(images_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                letter = file.split('.')[0].upper()
                if len(letter) == 1 and letter.isalpha():
                    available_letters.append(letter)
    
    return sorted(available_letters)

def create_sign_video(text: str, task_id: str, options: dict = None, app_context=None) -> Optional[str]:
    """Create sign language video from text"""
    try:
        if options is None:
            options = {}
            
        # Configuration - use app_context if provided, otherwise get from current_app
        if app_context:
            fps = options.get('video_fps', app_context.config.get('DEFAULT_FPS', 1))
            frame_duration = options.get('frame_duration', app_context.config.get('DEFAULT_FRAME_DURATION', 0.5))
            video_size = tuple(options.get('video_size', app_context.config.get('VIDEO_RESOLUTION', (640, 480))))
            images_dir = app_context.config.get('TEXT_TO_SIGN_IMAGES_DIR', 'TextToSignPipeline/sign_images')
            output_dir = app_context.config.get('TEXT_TO_SIGN_OUTPUT_DIR', 'TextToSignPipeline/output_videos')
        else:
            from flask import current_app
            fps = options.get('video_fps', current_app.config.get('DEFAULT_FPS', 1))
            frame_duration = options.get('frame_duration', current_app.config.get('DEFAULT_FRAME_DURATION', 0.5))
            video_size = tuple(options.get('video_size', current_app.config.get('VIDEO_RESOLUTION', (640, 480))))
            images_dir = get_sign_images_directory(app_context)
            output_dir = get_output_directory(app_context)
            
        background_color = tuple(options.get('background_color', (255, 255, 255)))
        show_text = options.get('show_text', True)
        filename = options.get('filename', f'sign_video_{task_id}.mp4')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean text
        clean_text = clean_text_for_signs(text)
        if not clean_text:
            raise ValueError("No valid letters found in text")
        
        processing_status.update_progress(task_id, 10, "Processing text...")
        
        # Create video
        output_path = os.path.join(output_dir, filename)
        
        # Try different codecs for better compatibility
        codecs_to_try = [
            ('mp4v', 'MP4V'),  # MP4 codec
            ('MJPG', 'MJPG'),  # MJPEG codec
            ('XVID', 'XVID'),  # XVID codec
            ('DIVX', 'DIVX'),  # DIVX codec
        ]
        
        out = None
        used_codec = None
        
        for codec_name, fourcc_code in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                out = cv2.VideoWriter(output_path, fourcc, fps, video_size)
                
                # Test if the VideoWriter was created successfully
                if out.isOpened():
                    used_codec = codec_name
                    logging.info(f"Successfully created video writer with codec: {codec_name}")
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                logging.warning(f"Failed to create video writer with codec {codec_name}: {e}")
                if out:
                    out.release()
                    out = None
        
        if out is None or not out.isOpened():
            raise Exception("Failed to create video writer with any available codec")
        
        processing_status.update_progress(task_id, 15, f"Using codec: {used_codec}")
        
        total_chars = len(clean_text.replace(' ', ''))
        processed_chars = 0
        
        for char in clean_text:
            if char == ' ':
                # Add blank frame for spaces
                blank_frame = np.full((*video_size[::-1], 3), background_color, dtype=np.uint8)
                for _ in range(int(fps * frame_duration)):
                    if out.isOpened():
                        out.write(blank_frame)
                    else:
                        raise Exception("Video writer closed unexpectedly")
                continue
            
            # Load sign image
            image_path = os.path.join(images_dir, f'{char}.png')
            if not os.path.exists(image_path):
                # Try jpg
                image_path = os.path.join(images_dir, f'{char}.jpg')
                if not os.path.exists(image_path):
                    logging.warning(f"Sign image not found for letter: {char}")
                    continue
            
            # Read and resize image
            img = cv2.imread(image_path)
            if img is None:
                logging.warning(f"Could not read image: {image_path}")
                continue
                
            img = cv2.resize(img, video_size)
            
            # Add text overlay if requested
            if show_text:
                cv2.putText(img, char, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            
            # Add frames based on duration
            for _ in range(int(fps * frame_duration)):
                if out.isOpened():
                    out.write(img)
                else:
                    raise Exception("Video writer closed unexpectedly during writing")
            
            processed_chars += 1
            progress = 15 + (processed_chars / total_chars) * 80
            processing_status.update_progress(task_id, int(progress), f"Processing letter: {char} ({processed_chars}/{total_chars})")
        
        # Release the video writer
        if out.isOpened():
            out.release()
        
        processing_status.update_progress(task_id, 95, "Finalizing video...")
        
        # Verify the video was created successfully
        if not os.path.exists(output_path):
            raise Exception("Video file was not created")
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise Exception("Video file is empty")
        
        logging.info(f"Video created successfully: {output_path} (size: {file_size} bytes)")
        
        processing_status.complete_task(task_id, {
            'video_path': output_path,
            'filename': filename,
            'text': text,
            'clean_text': clean_text,
            'duration': len(clean_text.replace(' ', '')) * frame_duration,
            'codec': used_codec,
            'file_size': file_size
        })
        return output_path
            
    except Exception as e:
        logging.error(f"Error creating sign video: {e}")
        processing_status.fail_task(task_id, str(e))
        return None

# Blueprint routes
@text_to_sign_bp.route('/convert', methods=['POST'])
def convert_text_to_sign():
    """Convert text to sign language video"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        from flask import current_app
        max_length = current_app.config.get('MAX_TEXT_LENGTH', 1000)
        if len(text) > max_length:
            return jsonify({'error': f'Text too long. Maximum {max_length} characters.'}), 400
        
        # Summarize if requested
        options = data.get('options', {})
        if options.get('summarize', False):
            text = text_summarizer.summarize(text, max_sentences=3)
        
        # Generate task ID
        task_id = f"text_to_sign_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Start processing in background
        processing_status.start_new_task(task_id)
        
        # Get proper app context for the background thread
        from flask import current_app
        app_context = current_app._get_current_object()
        
        def process_video():
            # Use with statement to properly manage app context in thread
            with app_context.app_context():
                create_sign_video(text, task_id, options)
        
        thread = threading.Thread(target=process_video)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Text-to-sign conversion started',
            'data': {
                'task_id': task_id,
                'status': 'processing',
                'status_url': f'/api/v1/text-to-sign/status/{task_id}',
                'input_text': text
            }
        })
        
    except Exception as e:
        logging.error(f"Error in convert_text_to_sign: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@text_to_sign_bp.route('/status/<task_id>')
def get_task_status(task_id):
    """Get task status"""
    try:
        status = processing_status.get_status(task_id)
        
        if status['status'] == 'not_found':
            return jsonify({'error': 'Task not found'}), 404
        
        response_data = {
            'success': True,
            'data': {
                'task_id': task_id,
                'status': status['status'],
                'progress': status['progress'],
                'message': status['message'],
                'completed': status['completed']
            }
        }
        
        if status['completed'] and status.get('result'):
            result = status['result']
            response_data['data']['result'] = {
                'filename': result['filename'],
                'download_url': f'/api/v1/text-to-sign/download/{task_id}',
                'text': result['text'],
                'duration': result.get('duration', 0)
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Error getting task status: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@text_to_sign_bp.route('/download/<task_id>')
def download_video(task_id):
    """Download generated video"""
    try:
        status = processing_status.get_status(task_id)
        
        if status['status'] != 'completed' or not status.get('result'):
            return jsonify({'error': 'Video not ready or not found'}), 404
        
        video_path = status['result']['video_path']
        filename = status['result']['filename']
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        return send_file(
            video_path,
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
        
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@text_to_sign_bp.route('/available-letters')
def get_available_letters_endpoint():
    """Get available sign letters"""
    try:
        from flask import current_app
        letters = get_available_letters(current_app)
        return jsonify({
            'success': True,
            'data': {
                'available_letters': letters,
                'count': len(letters)
            }
        })
    except Exception as e:
        logging.error(f"Error getting available letters: {e}")
        return jsonify({'error': 'Internal server error'}), 500
    

@text_to_sign_bp.route('/preview/<task_id>')
def preview_video(task_id):
    """Stream video for preview without download"""
    try:
        status = processing_status.get_status(task_id)
        
        if status['status'] != 'completed' or not status.get('result'):
            return jsonify({'error': 'Video not ready or not found'}), 404
        
        video_path = status['result']['video_path']
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        def generate():
            with open(video_path, 'rb') as f:
                while True:
                    chunk = f.read(1024)  # Read in 1KB chunks
                    if not chunk:
                        break
                    yield chunk
        
        return Response(
            stream_with_context(generate()),
            mimetype='video/mp4',
            headers={
                'Content-Disposition': 'inline',  # Display inline instead of download
                'Content-Type': 'video/mp4',
                'Accept-Ranges': 'bytes'
            }
        )
        
    except Exception as e:
        logging.error(f"Error streaming video: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@text_to_sign_bp.route('/health')
def health_check():
    """Health check for text-to-sign service"""
    try:
        from flask import current_app
        images_dir = get_sign_images_directory(current_app)
        output_dir = get_output_directory(current_app)
        
        return jsonify({
            'status': 'healthy',
            'service': 'text-to-sign',
            'images_directory': str(images_dir),
            'images_exist': os.path.exists(images_dir),
            'output_directory': str(output_dir),
            'available_letters': len(get_available_letters(current_app)),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500