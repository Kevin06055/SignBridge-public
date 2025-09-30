# app.py - Sign Language Conversion API with Quiz Features
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
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
import sys
from datetime import datetime, timedelta
from flask_cors import CORS

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Enable CORS with explicit headers
CORS(app, resources={r"/*": {
    "origins": "*",
    "allow_headers": ["Content-Type", "Authorization", "Accept"],
    "methods": ["GET", "POST", "OPTIONS"],
    "expose_headers": ["Content-Disposition"]
}})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Expose-Headers', 'Content-Disposition')
    return response

# Configuration
class Config:
    SIGN_IMAGES_DIR = "sign_images"  # Directory containing A.png, B.png, etc.
    OUTPUT_DIR = "output_videos"
    MAX_TEXT_LENGTH = 1000
    DEFAULT_FPS = 1  # Frames per second for video
    DEFAULT_FRAME_DURATION = 0.5  # Seconds per letter
    VIDEO_RESOLUTION = (640, 480)
    SUPPORTED_FORMATS = ['mp4', 'avi']
    # Quiz configuration
    QUIZ_SESSION_TIMEOUT = 1800  # 30 minutes
    MAX_QUIZ_OPTIONS = 4
    MIN_QUIZ_OPTIONS = 4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            logger.warning("NLTK stopwords not available, using basic set")
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
            logger.error(f"Summarization error: {e}")
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

class QuizGenerator:
    """Quiz generator for sign language learning"""
    
    def __init__(self, available_letters: List[str], images_dir: str):
        self.available_letters = available_letters
        self.images_dir = images_dir
        self.quiz_types = [
            'text_to_sign',    # "What is the sign for 'A'?" -> show 4 sign images
            'sign_to_text',    # Show a sign image -> "What letter is this?"
            'word_spelling',   # "Spell the word 'HELLO'" -> sequence of signs
        ]
    
    def generate_question(self, quiz_type: str = None, difficulty: str = 'easy') -> Dict[str, Any]:
        """Generate a quiz question based on type and difficulty"""
        if not quiz_type:
            quiz_type = random.choice(self.quiz_types)
        
        if quiz_type == 'text_to_sign':
            return self._generate_text_to_sign()
        elif quiz_type == 'sign_to_text':
            return self._generate_sign_to_text()
        elif quiz_type == 'word_spelling':
            return self._generate_word_spelling(difficulty)
        else:
            return self._generate_text_to_sign()
    
    def _generate_text_to_sign(self) -> Dict[str, Any]:
        """Generate 'What is the sign for letter X?' question"""
        correct_letter = random.choice(self.available_letters)
        
        # Get wrong options
        wrong_letters = [l for l in self.available_letters if l != correct_letter]
        wrong_choices = random.sample(wrong_letters, min(3, len(wrong_letters)))
        
        # Create options with image data
        all_options = wrong_choices + [correct_letter]
        random.shuffle(all_options)
        
        options = []
        for letter in all_options:
            image_path = self._get_image_path(letter)
            image_base64 = self._image_to_base64(image_path) if image_path else None
            options.append({
                'id': letter,
                'letter': letter,
                'image_url': f'/quiz-image/{letter}',
                'image_base64': image_base64
            })
        
        return {
            'id': f"tts_{int(time.time())}_{random.randint(1000, 9999)}",
            'type': 'text_to_sign',
            'question': f"What is the sign for letter '{correct_letter}'?",
            'question_text': correct_letter,
            'correct_answer': correct_letter,
            'options': options,
            'points': 10,
            'time_limit': 30
        }
    
    def _generate_sign_to_text(self) -> Dict[str, Any]:
        """Generate 'What letter is this sign?' question"""
        correct_letter = random.choice(self.available_letters)
        
        # Get wrong options
        wrong_letters = [l for l in self.available_letters if l != correct_letter]
        wrong_choices = random.sample(wrong_letters, min(3, len(wrong_letters)))
        
        all_options = wrong_choices + [correct_letter]
        random.shuffle(all_options)
        
        # Get the question image
        question_image_path = self._get_image_path(correct_letter)
        question_image_base64 = self._image_to_base64(question_image_path) if question_image_path else None
        
        options = [{'id': letter, 'text': letter} for letter in all_options]
        
        return {
            'id': f"stt_{int(time.time())}_{random.randint(1000, 9999)}",
            'type': 'sign_to_text',
            'question': "What letter does this sign represent?",
            'question_image_url': f'/quiz-image/{correct_letter}',
            'question_image_base64': question_image_base64,
            'correct_answer': correct_letter,
            'options': options,
            'points': 15,
            'time_limit': 20
        }
    
    def _generate_word_spelling(self, difficulty: str = 'easy') -> Dict[str, Any]:
        """Generate word spelling questions"""
        words = {
            'easy': ['CAT', 'DOG', 'BAT', 'HAT', 'PIG', 'BED', 'EGG'],
            'medium': ['HELLO', 'PHONE', 'TABLE', 'CHAIR', 'LIGHT'],
            'hard': ['FRIEND', 'SCHOOL', 'FAMILY', 'KITCHEN']
        }
        
        difficulty_words = words.get(difficulty, words['easy'])
        # Filter words to only include letters we have
        available_words = []
        for word in difficulty_words:
            if all(letter in self.available_letters for letter in word):
                available_words.append(word)
        
        if not available_words:
            # Fallback to simple words with available letters
            available_words = [''.join(random.sample(self.available_letters, k=min(3, len(self.available_letters)))) for _ in range(5)]
        
        correct_word = random.choice(available_words)
        
        # Create sequence of images for the word
        word_images = []
        for letter in correct_word:
            image_path = self._get_image_path(letter)
            image_base64 = self._image_to_base64(image_path) if image_path else None
            word_images.append({
                'letter': letter,
                'image_url': f'/quiz-image/{letter}',
                'image_base64': image_base64
            })
        
        # Generate wrong options
        wrong_words = [w for w in available_words if w != correct_word]
        wrong_choices = random.sample(wrong_words, min(3, len(wrong_words)))
        
        all_options = wrong_choices + [correct_word]
        random.shuffle(all_options)
        
        options = [{'id': word, 'text': word} for word in all_options]
        
        return {
            'id': f"spell_{int(time.time())}_{random.randint(1000, 9999)}",
            'type': 'word_spelling',
            'question': f"What word do these signs spell?",
            'question_images': word_images,
            'correct_answer': correct_word,
            'options': options,
            'points': 20,
            'time_limit': 45
        }
    
    def _get_image_path(self, letter: str) -> Optional[str]:
        """Get the full path to a letter's image"""
        for ext in ['png', 'jpg', 'jpeg']:
            path = os.path.join(self.images_dir, f"{letter}.{ext}")
            if os.path.exists(path):
                return path
        return None
    
    def _image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert image to base64 string"""
        if not image_path or not os.path.exists(image_path):
            return None
        
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                # Get file extension
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
                return f"data:{mime_type};base64,{img_base64}"
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return None

class QuizSession:
    """Manage individual quiz sessions"""
    
    def __init__(self, session_id: str, player_name: str = "Anonymous"):
        self.session_id = session_id
        self.player_name = player_name
        self.start_time = datetime.now()
        self.current_question = None
        self.questions_answered = 0
        self.correct_answers = 0
        self.total_score = 0
        self.question_history = []
        self.is_active = True
        self.last_activity = datetime.now()
    
    def add_question(self, question: Dict[str, Any]):
        """Add a new question to the session"""
        self.current_question = question
        self.last_activity = datetime.now()
    
    def submit_answer(self, answer: str, time_taken: float) -> Dict[str, Any]:
        """Submit an answer and get results"""
        if not self.current_question:
            return {'error': 'No active question'}
        
        is_correct = answer == self.current_question['correct_answer']
        points_earned = 0
        
        if is_correct:
            self.correct_answers += 1
            # Calculate points based on time and difficulty
            base_points = self.current_question.get('points', 10)
            time_bonus = max(0, (self.current_question.get('time_limit', 30) - time_taken) / 2)
            points_earned = int(base_points + time_bonus)
            self.total_score += points_earned
        
        # Record the question and answer
        result_question = self.current_question.copy()
        self.question_history.append({
            'question': result_question,
            'user_answer': answer,
            'correct_answer': self.current_question['correct_answer'],
            'is_correct': is_correct,
            'points_earned': points_earned,
            'time_taken': time_taken,
            'timestamp': datetime.now().isoformat()
        })
        
        self.questions_answered += 1
        correct_answer = self.current_question['correct_answer']
        self.current_question = None
        self.last_activity = datetime.now()
        
        return {
            'is_correct': is_correct,
            'correct_answer': correct_answer,
            'points_earned': points_earned,
            'total_score': self.total_score,
            'questions_answered': self.questions_answered,
            'accuracy': (self.correct_answers / self.questions_answered * 100) if self.questions_answered > 0 else 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        duration = datetime.now() - self.start_time
        return {
            'session_id': self.session_id,
            'player_name': self.player_name,
            'questions_answered': self.questions_answered,
            'correct_answers': self.correct_answers,
            'total_score': self.total_score,
            'accuracy': (self.correct_answers / self.questions_answered * 100) if self.questions_answered > 0 else 0,
            'duration_seconds': int(duration.total_seconds()),
            'average_time_per_question': int(duration.total_seconds() / self.questions_answered) if self.questions_answered > 0 else 0,
            'is_active': self.is_active
        }

class SignLanguageProcessor:
    """Main processor for converting text to sign language video"""
    
    def __init__(self, config: Config):
        self.config = config
        self.summarizer = TextSummarizer()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(self.config.SIGN_IMAGES_DIR):
            raise FileNotFoundError(f"Sign images directory not found: {self.config.SIGN_IMAGES_DIR}")
    
    def process_text_to_video(self, text: str, output_filename: str = None, 
                            summarize: bool = True, fps: int = None, task_id: str = None) -> str:
        """Complete pipeline: text -> summary -> words -> letters -> video"""
        try:
            # Step 1: Summarize text
            if task_id:
                processing_status.update_progress(task_id, 10, "Summarizing text...")
            
            if summarize and len(text) > 100:
                processed_text = self.summarizer.summarize(text)
                logger.info(f"Summarized text: {processed_text}")
            else:
                processed_text = text
            
            # Step 2: Split into words and then letters
            if task_id:
                processing_status.update_progress(task_id, 20, "Extracting letters...")
            
            letters = self._extract_letters(processed_text)
            logger.info(f"Extracted {len(letters)} letters")
            
            # Step 3: Fetch corresponding images
            if task_id:
                processing_status.update_progress(task_id, 30, "Fetching sign images...")
            
            image_paths = self._fetch_letter_images(letters)
            
            # Step 4: Generate video
            if task_id:
                processing_status.update_progress(task_id, 40, "Preparing to generate video...")
            
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"sign_video_{timestamp}.mp4"
            
            video_path = self._generate_video(image_paths, output_filename, 
                                            fps or self.config.DEFAULT_FPS, task_id)
            
            # Final update
            if task_id:
                processing_status.update_progress(task_id, 100, "Video generated successfully")
            
            return video_path
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            if task_id:
                processing_status.fail_task(task_id, str(e))
            raise
    
    def _extract_letters(self, text: str) -> List[str]:
        """Extract letters from text, handling spaces and punctuation"""
        # Remove extra whitespace and convert to uppercase
        text = re.sub(r'\s+', ' ', text.strip()).upper()
        
        letters = []
        for char in text:
            if char.isalpha():
                letters.append(char)
            elif char == ' ':
                letters.append('SPACE')  # Special token for space
            # Skip punctuation for now
        
        return letters
    
    def _fetch_letter_images(self, letters: List[str]) -> List[str]:
        """Fetch image paths for each letter"""
        image_paths = []
        missing_letters = []
        
        for letter in letters:
            if letter == 'SPACE':
                # Create a blank image for space or skip
                image_paths.append(None)  # Will handle in video generation
            else:
                image_path = os.path.join(self.config.SIGN_IMAGES_DIR, f"{letter}.jpg")
                
                # Also try PNG format
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.config.SIGN_IMAGES_DIR, f"{letter}.png")
                
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                else:
                    missing_letters.append(letter)
                    logger.warning(f"Image not found for letter: {letter}")
                    image_paths.append(None)
        
        if missing_letters:
            logger.warning(f"Missing images for letters: {missing_letters}")
        
        return image_paths
    
    def _generate_video(self, image_paths: List[str], output_filename: str, fps: int, task_id: str = None) -> str:
        """Generate video from image sequence"""
        output_path = os.path.join(self.config.OUTPUT_DIR, output_filename)
        
        # Filter out None values and get valid images
        valid_images = [path for path in image_paths if path is not None]
        
        if not valid_images:
            raise ValueError("No valid images found to create video")
        
        # Get video dimensions from first image
        first_img = cv2.imread(valid_images[0])
        if first_img is None:
            raise ValueError(f"Could not read first image: {valid_images[0]}")
        
        height, width, layers = first_img.shape
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            total_frames = len(image_paths)
            
            for i, image_path in enumerate(image_paths):
                # Update progress (from 40% to 90%)
                if task_id and total_frames > 0:
                    progress = 40 + int((i / total_frames) * 50)
                    processing_status.update_progress(
                        task_id, 
                        progress, 
                        f"Generating video: {i+1}/{total_frames} frames"
                    )
                
                if image_path is None:
                    # Handle space - add blank frame or pause
                    blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    for _ in range(int(fps * 0.3)):  # 0.3 second pause for space
                        video_writer.write(blank_frame)
                else:
                    img = cv2.imread(image_path)
                    if img is not None:
                        # Resize image to match video dimensions
                        img_resized = cv2.resize(img, (width, height))
                        
                        # Write frame multiple times based on desired duration
                        frame_count = int(fps * self.config.DEFAULT_FRAME_DURATION)
                        for _ in range(max(1, frame_count)):
                            video_writer.write(img_resized)
                    else:
                        logger.warning(f"Could not read image: {image_path}")
            
            # Final video processing
            if task_id:
                processing_status.update_progress(task_id, 95, "Finalizing video...")
        
        finally:
            video_writer.release()
        
        logger.info(f"Video generated: {output_path}")
        return output_path
    
    def get_available_letters(self) -> List[str]:
        """Get list of available letter images"""
        available = []
        for ext in ['png', 'jpg', 'jpeg']:
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                image_path = os.path.join(self.config.SIGN_IMAGES_DIR, f"{letter}.{ext}")
                if os.path.exists(image_path):
                    available.append(letter)
                    break
        return sorted(list(set(available)))

# Initialize components
config = Config()
processor = SignLanguageProcessor(config)
quiz_generator = QuizGenerator(processor.get_available_letters(), config.SIGN_IMAGES_DIR)

# Utility functions for quiz management
def cleanup_expired_sessions():
    """Remove expired quiz sessions"""
    global quiz_sessions
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session in quiz_sessions.items():
        if (current_time - session.last_activity).seconds > config.QUIZ_SESSION_TIMEOUT:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del quiz_sessions[session_id]

def generate_session_id() -> str:
    """Generate unique session ID"""
    return f"quiz_{int(time.time())}_{random.randint(10000, 99999)}"

# ORIGINAL API ROUTES
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed diagnostics"""
    cleanup_expired_sessions()
    
    # Check if output directory exists and is writable
    output_dir_status = {
        'exists': os.path.exists(config.OUTPUT_DIR),
        'writable': os.access(config.OUTPUT_DIR, os.W_OK) if os.path.exists(config.OUTPUT_DIR) else False,
        'path': config.OUTPUT_DIR,
        'absolute_path': os.path.abspath(config.OUTPUT_DIR)
    }
    
    # Check if sign images directory exists and is readable
    sign_dir_status = {
        'exists': os.path.exists(config.SIGN_IMAGES_DIR),
        'readable': os.access(config.SIGN_IMAGES_DIR, os.R_OK) if os.path.exists(config.SIGN_IMAGES_DIR) else False,
        'path': config.SIGN_IMAGES_DIR,
        'absolute_path': os.path.abspath(config.SIGN_IMAGES_DIR),
    }
    
    # Add sign images information if directory exists
    if sign_dir_status['exists']:
        sign_files = [f for f in os.listdir(config.SIGN_IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        sign_dir_status.update({
            'file_count': len(sign_files),
            'files': sign_files[:10],  # List first 10 files
            'has_more_files': len(sign_files) > 10
        })
    
    # List available videos
    available_videos = []
    if output_dir_status['exists'] and os.access(config.OUTPUT_DIR, os.R_OK):
        try:
            video_files = [f for f in os.listdir(config.OUTPUT_DIR) if f.endswith(('.mp4', '.avi'))]
            available_videos = [{'name': f, 'url': f'/view/{f}'} for f in video_files[:5]]  # Limit to 5 videos
        except Exception as e:
            logger.error(f"Error listing videos: {e}")
    
    # Get processor status
    processor_status = {
        'available': processor is not None,
        'quiz_available': quiz_generator is not None,
        'available_letters': processor.get_available_letters() if processor is not None else [],
        'current_working_directory': os.getcwd(),
        'python_version': sys.version
    }
    
    return jsonify({
        'status': 'healthy',
        'processor': processor_status,
        'directories': {
            'output': output_dir_status,
            'sign_images': sign_dir_status
        },
        'available_videos': available_videos,
        'active_quiz_sessions': len(quiz_sessions),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/convert', methods=['POST'])
def convert_text_to_sign():
    """Convert text to sign language video"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text field is required'}), 400
        
        text = data['text']
        if len(text) > config.MAX_TEXT_LENGTH:
            return jsonify({'error': f'Text too long. Maximum {config.MAX_TEXT_LENGTH} characters'}), 400
        
        # Optional parameters
        summarize = data.get('summarize', True)
        fps = data.get('fps', config.DEFAULT_FPS)
        output_filename = data.get('filename')
        
        # Create a task ID for tracking progress
        task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(text)}"
        processing_status.start_new_task(task_id)
        
        # Start a background thread for processing
        def process_in_background():
            try:
                video_path = processor.process_text_to_video(
                    text=text,
                    output_filename=output_filename,
                    summarize=summarize,
                    fps=fps,
                    task_id=task_id
                )
                
                # Log the video info for debugging
                video_filename = os.path.basename(video_path)
                logger.info(f"Generated video: {video_filename}")
                
                result = {
                    'success': True,
                    'video_path': video_path,
                    'video_url': f'/view/{video_filename}',
                    'download_url': f'/download/{video_filename}',
                    'message': 'Video generated successfully'
                }
                
                logger.info(f"Video URLs: view={result['video_url']}, download={result['download_url']}")
                
                processing_status.complete_task(task_id, result)
            except Exception as e:
                logger.error(f"Background processing error: {e}")
                processing_status.fail_task(task_id, str(e))
        
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'status': 'processing',
            'message': 'Processing started'
        })
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get the status of a processing task"""
    status = processing_status.get_status(task_id)
    
    if status.get('status') == 'not_found':
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(status)

@app.route('/download/<filename>', methods=['GET'])
def download_video(filename):
    """Download generated video"""
    try:
        video_path = os.path.join(config.OUTPUT_DIR, filename)
        logger.info(f"Download requested for: {filename}, Full path: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"File not found for download: {video_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Determine MIME type based on file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == '.mp4':
            mimetype = 'video/mp4'
        elif file_ext == '.avi':
            mimetype = 'video/x-msvideo'
        elif file_ext == '.webm':
            mimetype = 'video/webm'
        else:
            mimetype = 'application/octet-stream'  # Default to binary for unknown types
            
        logger.info(f"Serving download with MIME type: {mimetype}")
        
        # Set proper filename in Content-Disposition header
        return send_file(
            video_path, 
            as_attachment=True, 
            download_name=filename,
            mimetype=mimetype
        )
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/view/<filename>', methods=['GET'])
def view_video(filename):
    """View video in browser (without downloading)"""
    try:
        video_path = os.path.join(config.OUTPUT_DIR, filename)
        logger.info(f"Requested video: {filename}, Full path: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"File not found: {video_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Log file information
        file_size = os.path.getsize(video_path)
        logger.info(f"Serving video: {filename}, Size: {file_size} bytes")
        
        # Determine MIME type based on file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == '.mp4':
            mimetype = 'video/mp4'
        elif file_ext == '.avi':
            mimetype = 'video/x-msvideo'
        elif file_ext == '.webm':
            mimetype = 'video/webm'
        else:
            mimetype = 'video/mp4'  # Default to mp4 if unknown
            
        logger.info(f"Serving video with MIME type: {mimetype}")
        
        # Add Cache-Control to prevent caching issues
        response = send_file(video_path, as_attachment=False, mimetype=mimetype)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
        
    except Exception as e:
        logger.error(f"View error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/preview', methods=['POST'])
def preview_processing():
    """Preview text processing without generating video"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text field is required'}), 400
        
        text = data['text']
        summarize = data.get('summarize', True)
        
        # Process text
        if summarize and len(text) > 100:
            summarized = processor.summarizer.summarize(text)
        else:
            summarized = text
        
        letters = processor._extract_letters(summarized)
        available_letters = processor.get_available_letters()
        missing_letters = [l for l in letters if l not in available_letters and l != 'SPACE']
        
        return jsonify({
            'original_text': text,
            'summarized_text': summarized,
            'letters': letters,
            'letter_count': len(letters),
            'missing_letters': missing_letters,
            'available_letters': available_letters
        })
        
    except Exception as e:
        logger.error(f"Preview error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/letters', methods=['GET'])
def get_available_letters():
    """Get available letter images"""
    if processor is None:
        return jsonify({
            'error': 'Sign language processor is not available',
            'available_letters': []
        }), 500
        
    return jsonify({
        'available_letters': processor.get_available_letters()
    })

# NEW QUIZ API ROUTES
@app.route('/quiz/start', methods=['POST'])
def start_quiz():
    """Start a new quiz session"""
    # Check if quiz generator is available
    if quiz_generator is None:
        return jsonify({
            'error': 'Quiz functionality is not available. Sign language processor failed to initialize.',
            'details': 'Please check that the sign_images directory exists and contains image files.'
        }), 500
        
    try:
        data = request.get_json() or {}
        player_name = data.get('player_name', 'Anonymous')
        
        session_id = generate_session_id()
        session = QuizSession(session_id, player_name)
        quiz_sessions[session_id] = session
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'player_name': player_name,
            'message': 'Quiz session started successfully'
        })
        
    except Exception as e:
        logger.error(f"Quiz start error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/quiz/<session_id>/question', methods=['GET'])
def get_quiz_question(session_id):
    """Get a new quiz question"""
    # Check if quiz generator is available
    if quiz_generator is None:
        return jsonify({
            'error': 'Quiz functionality is not available. Sign language processor failed to initialize.',
            'details': 'Please check that the sign_images directory exists and contains image files.'
        }), 500
        
    try:
        cleanup_expired_sessions()
        
        if session_id not in quiz_sessions:
            return jsonify({'error': 'Invalid session ID'}), 404
        
        session = quiz_sessions[session_id]
        if not session.is_active:
            return jsonify({'error': 'Session is not active'}), 400
        
        # Get query parameters
        quiz_type = request.args.get('type', None)
        difficulty = request.args.get('difficulty', 'easy')
        
        # Generate question
        question = quiz_generator.generate_question(quiz_type, difficulty)
        session.add_question(question)
        
        return jsonify({
            'success': True,
            'question': question,
            'session_stats': session.get_stats()
        })
        
    except Exception as e:
        logger.error(f"Quiz question error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/quiz/<session_id>/answer', methods=['POST'])
def submit_quiz_answer(session_id):
    """Submit an answer to the current question"""
    try:
        cleanup_expired_sessions()
        
        if session_id not in quiz_sessions:
            return jsonify({'error': 'Invalid session ID'}), 404
        
        data = request.get_json()
        if not data or 'answer' not in data:
            return jsonify({'error': 'Answer is required'}), 400
        
        answer = data['answer']
        time_taken = data.get('time_taken', 0)
        
        session = quiz_sessions[session_id]
        result = session.submit_answer(answer, time_taken)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            'result': result,
            'session_stats': session.get_stats()
        })
        
    except Exception as e:
        logger.error(f"Quiz answer error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/quiz/<session_id>/stats', methods=['GET'])
def get_quiz_stats(session_id):
    """Get quiz session statistics"""
    try:
        cleanup_expired_sessions()
        
        if session_id not in quiz_sessions:
            return jsonify({'error': 'Invalid session ID'}), 404
        
        session = quiz_sessions[session_id]
        return jsonify({
            'success': True,
            'stats': session.get_stats(),
            'history': session.question_history[-10:]  # Last 10 questions only
        })
        
    except Exception as e:
        logger.error(f"Quiz stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/quiz/<session_id>/end', methods=['POST'])
def end_quiz(session_id):
    """End a quiz session"""
    try:
        cleanup_expired_sessions()
        
        if session_id not in quiz_sessions:
            return jsonify({'error': 'Invalid session ID'}), 404
        
        session = quiz_sessions[session_id]
        session.is_active = False
        
        # Add to leaderboard
        final_stats = session.get_stats()
        quiz_leaderboard.append({
            'player_name': session.player_name,
            'score': session.total_score,
            'accuracy': final_stats['accuracy'],
            'questions_answered': session.questions_answered,
            'duration': final_stats['duration_seconds'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Sort leaderboard by score
        quiz_leaderboard.sort(key=lambda x: x['score'], reverse=True)
        # Keep only top 100
        if len(quiz_leaderboard) > 100:
            quiz_leaderboard[:] = quiz_leaderboard[:100]
        
        # Find rank
        rank = None
        for i, entry in enumerate(quiz_leaderboard):
            if (entry['player_name'] == session.player_name and 
                entry['score'] == session.total_score and
                entry['timestamp'] == quiz_leaderboard[-1]['timestamp']):
                rank = i + 1
                break
        
        return jsonify({
            'success': True,
            'final_stats': final_stats,
            'rank': rank,
            'message': 'Quiz ended successfully'
        })
        
    except Exception as e:
        logger.error(f"Quiz end error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/quiz/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get the quiz leaderboard"""
    try:
        limit = min(int(request.args.get('limit', 10)), 50)
        return jsonify({
            'success': True,
            'leaderboard': quiz_leaderboard[:limit],
            'total_players': len(quiz_leaderboard)
        })
        
    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/quiz-image/<letter>', methods=['GET'])
def get_quiz_letter_image(letter):
    """Serve letter images for quiz questions"""
    # Check if processor is available
    if processor is None:
        return jsonify({
            'error': 'Sign language processor is not available',
            'details': 'Sign images directory could not be found'
        }), 500
        
    try:
        letter = letter.upper()
        available_letters = processor.get_available_letters()
        
        if letter not in available_letters:
            return jsonify({'error': 'Letter image not found'}), 404
        
        # Find the image file
        for ext in ['png', 'jpg', 'jpeg']:
            image_path = os.path.join(config.SIGN_IMAGES_DIR, f"{letter}.{ext}")
            if os.path.exists(image_path):
                return send_file(image_path)
        
        return jsonify({'error': 'Image file not found'}), 404
        
    except Exception as e:
        logger.error(f"Quiz image serve error: {e}")
        return jsonify({'error': str(e)}), 500

# DEBUG ROUTES
@app.route('/debug/videos', methods=['GET'])
def debug_videos():
    """Debug endpoint to list and test video access"""
    try:
        if not os.path.exists(config.OUTPUT_DIR):
            return jsonify({
                'error': 'Output directory does not exist',
                'directory': config.OUTPUT_DIR
            }), 404
        
        videos = []
        for filename in os.listdir(config.OUTPUT_DIR):
            if filename.endswith(('.mp4', '.avi')):
                file_path = os.path.join(config.OUTPUT_DIR, filename)
                videos.append({
                    'filename': filename,
                    'size': os.path.getsize(file_path),
                    'created': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                    'view_url': f'/view/{filename}',
                    'download_url': f'/download/{filename}'
                })
        
        return jsonify({
            'output_dir': config.OUTPUT_DIR,
            'video_count': len(videos),
            'videos': videos,
            'readable': os.access(config.OUTPUT_DIR, os.R_OK),
            'writable': os.access(config.OUTPUT_DIR, os.W_OK)
        })
    
    except Exception as e:
        logger.error(f"Debug videos error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug/test-video', methods=['GET'])
def debug_test_video():
    """Generate a test video to verify video creation and serving"""
    try:
        # Create a simple test video with OpenCV
        output_filename = f"test_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(config.OUTPUT_DIR, output_filename)
        
        # Create a blank video
        width, height = 640, 480
        fps = 30
        duration = 3  # seconds
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Generate frames with text
        frames = fps * duration
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i in range(frames):
            # Create a gradient background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw frame number
            text = f"Test Frame {i+1}/{frames}"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, height - 20), font, 0.5, (200, 200, 200), 1)
            
            video_writer.write(frame)
        
        video_writer.release()
        
        return jsonify({
            'success': True,
            'message': 'Test video created successfully',
            'filename': output_filename,
            'path': output_path,
            'view_url': f'/view/{output_filename}',
            'download_url': f'/download/{output_filename}'
        })
        
    except Exception as e:
        logger.error(f"Debug test video error: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create sample directory structure info
    print("Sign Language Conversion & Quiz API")
    print("=" * 45)
    print(f"Sign images directory: {config.SIGN_IMAGES_DIR}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Available letters: {processor.get_available_letters()}")
    print("\nOriginal API Endpoints:")
    print("- POST /convert - Convert text to sign language video")
    print("- GET /status/<task_id> - Get processing status")
    print("- POST /preview - Preview text processing")
    print("- GET /letters - Get available letters")
    print("- GET /health - Health check")
    print("- GET /download/<filename> - Download video")
    print("- GET /view/<filename> - View video in browser")
    print("\nNEW Quiz API Endpoints:")
    print("- POST /quiz/start - Start a new quiz session")
    print("- GET /quiz/<session_id>/question - Get a quiz question")
    print("- POST /quiz/<session_id>/answer - Submit quiz answer")
    print("- GET /quiz/<session_id>/stats - Get quiz statistics")
    print("- POST /quiz/<session_id>/end - End quiz session")
    print("- GET /quiz/leaderboard - Get leaderboard")
    print("- GET /quiz-image/<letter> - Get letter image for quiz")
    print("\nDebug Endpoints:")
    print("- GET /debug/videos - List all generated videos")
    print("- GET /debug/test-video - Create test video")
    print("\nStarting server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
