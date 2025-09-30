"""
Speech-to-Sign Blueprint with OpenAI Whisper
Handles audio file upload, speech recognition using Whisper, and conversion to sign language
"""

from flask import Blueprint, request, jsonify, send_file
import logging
from datetime import datetime
import time
import random
import threading
import whisper
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename

# Import text-to-sign functionality
from .text_to_sign import create_sign_video, processing_status, text_summarizer, clean_text_for_signs

# Create blueprint
speech_to_sign_bp = Blueprint('speech_to_sign', __name__)

# Audio file configurations
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'webm', 'mp4', 'wma', 'aac'}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB limit (Whisper API limit)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class SpeechToSignProcessor:
    """Processor for speech-to-sign conversion with Whisper"""
    
    def __init__(self, model_size="base"):
        self.active_tasks = {}
        self.lock = threading.Lock()
        try:
            # Load Whisper model once when processor is initialized
            self.whisper_model = whisper.load_model(model_size)
            self.model_loaded = True
            logging.info(f"Whisper model '{model_size}' loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            self.model_loaded = False
    
    def transcribe_audio(self, audio_path: str, options: dict = None) -> str:
        """Transcribe audio file using Whisper"""
        if not self.model_loaded:
            raise Exception("Whisper model not loaded")
        
        if options is None:
            options = {}
        
        # Whisper transcription options
        whisper_options = {
            'language': options.get('language'),  # None = auto-detect
            'task': options.get('task', 'transcribe'),  # 'transcribe' or 'translate'
            'fp16': options.get('fp16', True),  # Use FP16 for faster processing
            'verbose': options.get('verbose', False)
        }
        
        # Remove None values
        whisper_options = {k: v for k, v in whisper_options.items() if v is not None}
        
        try:
            result = self.whisper_model.transcribe(audio_path, **whisper_options)
            return result['text'].strip()
        except Exception as e:
            logging.error(f"Whisper transcription failed: {e}")
            raise Exception(f"Audio transcription failed: {str(e)}")
    
    def process_audio_to_sign(self, audio_path: str, task_id: str, options: dict = None):
        """Process audio file and convert to sign language video"""
        try:
            if options is None:
                options = {}
            
            # Update processing status
            processing_status.update_progress(task_id, 5, "Starting audio transcription...")
            
            # Transcribe audio using Whisper
            processing_status.update_progress(task_id, 15, "Transcribing audio with Whisper AI...")
            transcribed_text = self.transcribe_audio(audio_path, options.get('whisper_options', {}))
            
            if not transcribed_text:
                processing_status.fail_task(task_id, "No speech detected in audio file")
                return
            
            processing_status.update_progress(task_id, 30, f"Transcription complete: {transcribed_text[:50]}...")
            
            # Clean and validate text for signs
            clean_text = clean_text_for_signs(transcribed_text)
            if not clean_text:
                processing_status.fail_task(task_id, "No valid letters found in transcribed speech")
                return
            
            processing_status.update_progress(task_id, 40, "Converting transcribed text to sign language...")
            
            # Store transcription info in options so it gets passed to create_sign_video
            options.update({
                'transcribed_text': transcribed_text,
                'original_audio_file': True
            })
            
            # Use existing text-to-sign functionality
            from flask import current_app
            video_path = create_sign_video(clean_text, task_id, options, current_app)
            
            if video_path:
                # The transcription info is now handled by the processing_status system
                # Get the current result and update it
                current_status = processing_status.get_status(task_id)
                if current_status.get('result'):
                    # Update the existing result with transcription info
                    result = current_status['result']
                    result.update({
                        'transcribed_text': transcribed_text,
                        'clean_text': clean_text
                    })
                    processing_status.complete_task(task_id, result)
                
                processing_status.update_progress(task_id, 100, "Audio-to-sign conversion completed")
            else:
                processing_status.fail_task(task_id, "Failed to generate sign language video")
                
        except Exception as e:
            logging.error(f"Error in audio-to-sign processing: {e}")
            processing_status.fail_task(task_id, str(e))
        finally:
            # Clean up audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                logging.warning(f"Failed to clean up audio file: {e}")

# Initialize processor with base model (you can change to 'small', 'medium', 'large')
speech_processor = SpeechToSignProcessor(model_size="base")

@speech_to_sign_bp.route('/upload-audio', methods=['POST'])
def upload_and_convert_audio():
    """Upload audio file and convert to sign language video"""
    try:
        # Check if model is loaded
        if not speech_processor.model_loaded:
            return jsonify({
                'success': False,
                'error': 'Whisper model not available'
            }), 503
        
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Validate file
        if not allowed_file(audio_file.filename):
            return jsonify({
                'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Check file size
        audio_file.seek(0, os.SEEK_END)
        file_size = audio_file.tell()
        audio_file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'error': f'File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB'
            }), 400
        
        # Get additional options
        data = request.form.to_dict()
        options = {
            'show_text': data.get('show_text', 'true').lower() == 'true',
            'frame_duration': float(data.get('frame_duration', 1.0)),
            'video_fps': int(data.get('video_fps', 1)),
            'filename': data.get('filename', f'audio_to_sign_{int(time.time())}.mp4'),
            'summarize': data.get('summarize', 'false').lower() == 'true'
        }
        
        # Whisper-specific options
        whisper_options = {
            'language': data.get('language'),  # None for auto-detect
            'task': data.get('task', 'transcribe'),
            'fp16': data.get('fp16', 'true').lower() == 'true'
        }
        options['whisper_options'] = whisper_options
        
        # Generate task ID
        task_id = f"audio_to_sign_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Save audio file temporarily
        file_extension = secure_filename(audio_file.filename).rsplit('.', 1)[1].lower()
        temp_filename = f"{uuid.uuid4().hex}.{file_extension}"
        
        from flask import current_app
        upload_dir = current_app.config.get('UPLOAD_FOLDER', tempfile.gettempdir())
        os.makedirs(upload_dir, exist_ok=True)
        
        audio_path = os.path.join(upload_dir, temp_filename)
        audio_file.save(audio_path)
        
        # Create processing task
        processing_status.start_new_task(task_id)
        
        # Save the app context for the background thread
        app_context = current_app._get_current_object()
        
        def process_task():
            with app_context.app_context():
                speech_processor.process_audio_to_sign(audio_path, task_id, options)
        
        thread = threading.Thread(target=process_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Audio uploaded successfully, processing started',
            'data': {
                'task_id': task_id,
                'status': 'processing',
                'status_url': f'/api/v1/speech-to-sign/status/{task_id}',
                'download_url': f'/api/v1/speech-to-sign/download/{task_id}',
                'file_size': file_size,
                'file_type': file_extension.upper(),
                'estimated_duration': '2-5 minutes'
            }
        })
        
    except Exception as e:
        logging.error(f"Error in upload_and_convert_audio: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@speech_to_sign_bp.route('/convert-text', methods=['POST'])
def convert_text_to_sign():
    """Convert text directly to sign language video (legacy endpoint)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract text
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        from flask import current_app
        max_length = current_app.config.get('MAX_TEXT_LENGTH', 1000)
        if len(text) > max_length:
            return jsonify({'error': f'Text too long. Maximum {max_length} characters.'}), 400
        
        # Get options
        options = data.get('options', {})
        options.update({
            'show_text': data.get('show_text', True),
            'frame_duration': data.get('frame_duration', 1.0),
            'video_fps': data.get('video_fps', 1),
            'filename': data.get('filename', f'text_to_sign_{int(time.time())}.mp4')
        })
        
        # Summarize if requested
        if options.get('summarize', False):
            text = text_summarizer.summarize(text, max_sentences=3)
        
        # Generate task ID
        task_id = f"text_to_sign_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create processing task
        processing_status.start_new_task(task_id)
        
        # Save the app context for the background thread
        from flask import current_app
        app_context = current_app._get_current_object()
        
        def process_task():
            with app_context.app_context():
                try:
                    processing_status.update_progress(task_id, 10, "Processing text input...")
                    clean_text = clean_text_for_signs(text)
                    if not clean_text:
                        processing_status.fail_task(task_id, "No valid letters found in text")
                        return
                    
                    processing_status.update_progress(task_id, 20, "Converting text to sign language...")
                    result = create_sign_video(clean_text, task_id, options, current_app)
                    
                    if result:
                        processing_status.update_progress(task_id, 100, "Text-to-sign conversion completed")
                    else:
                        processing_status.fail_task(task_id, "Failed to generate sign language video")
                except Exception as e:
                    processing_status.fail_task(task_id, str(e))
        
        thread = threading.Thread(target=process_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Text-to-sign conversion started',
            'data': {
                'task_id': task_id,
                'status': 'processing',
                'status_url': f'/api/v1/speech-to-sign/status/{task_id}',
                'download_url': f'/api/v1/speech-to-sign/download/{task_id}',
                'input_text': text,
                'clean_text': clean_text_for_signs(text)
            }
        })
        
    except Exception as e:
        logging.error(f"Error in convert_text_to_sign: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@speech_to_sign_bp.route('/status/<task_id>')
def get_task_status(task_id):
    """Get task status for both audio and text processing"""
    try:
        status = processing_status.get_status(task_id)
        
        if status['status'] == 'not_found':
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
        
        response_data = {
            'success': True,
            'data': {
                'task_id': task_id,
                'status': status['status'],
                'progress': status['progress'],
                'message': status['message'],
                'completed': status['completed'],
                'task_type': 'audio-to-sign' if 'audio' in task_id else 'text-to-sign',
                'created_at': status.get('start_time'),
                'updated_at': datetime.now().isoformat()
            }
        }
        
        # Add result information if completed
        if status['completed'] and status.get('result'):
            result = status['result']
            response_data['data']['result'] = {
                'filename': result['filename'],
                'download_url': f'/api/v1/speech-to-sign/download/{task_id}',
                'duration': result.get('duration', 0),
                'video_path': result['video_path']
            }
            
            # Add transcription info for audio tasks
            if 'transcribed_text' in result:
                response_data['data']['result'].update({
                    'transcribed_text': result['transcribed_text'],
                    'clean_text': result['clean_text']
                })
            else:
                response_data['data']['result'].update({
                    'text': result.get('text', ''),
                    'clean_text': result.get('clean_text', '')
                })
        
        # Add error information if failed
        if status['status'] == 'failed':
            response_data['data']['error'] = status['message']
        
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Error getting task status: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@speech_to_sign_bp.route('/download/<task_id>')
def download_video(task_id):
    """Download generated sign language video"""
    try:
        status = processing_status.get_status(task_id)
        
        if status['status'] != 'completed' or not status.get('result'):
            return jsonify({
                'success': False,
                'error': 'Video not ready or not found'
            }), 404
        
        video_path = status['result']['video_path']
        filename = status['result']['filename']
        
        if not os.path.exists(video_path):
            return jsonify({
                'success': False,
                'error': 'Video file not found'
            }), 404
        
        return send_file(
            video_path,
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
        
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@speech_to_sign_bp.route('/available-letters')
def get_available_letters():
    """Get available sign letters"""
    try:
        from .text_to_sign import get_available_letters
        
        from flask import current_app
        letters = get_available_letters(current_app)
        return jsonify({
            'success': True,
            'data': {
                'available_letters': letters,
                'count': len(letters),
                'service': 'speech-to-sign-whisper'
            }
        })
        
    except Exception as e:
        logging.error(f"Error getting available letters: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@speech_to_sign_bp.route('/supported-formats')
def get_supported_formats():
    """Get supported audio formats and configurations"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'supported_formats': list(ALLOWED_EXTENSIONS),
                'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),
                'whisper_models': ['tiny', 'base', 'small', 'medium', 'large'],
                'current_model': speech_processor.whisper_model.dims.n_mels if speech_processor.model_loaded else 'Not loaded',
                'supported_languages': [
                    'auto-detect', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi'
                ],
                'tasks': ['transcribe', 'translate']
            }
        })
        
    except Exception as e:
        logging.error(f"Error getting supported formats: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@speech_to_sign_bp.route('/health')
def health_check():
    """Health check for speech-to-sign service"""
    try:
        from .text_to_sign import get_sign_images_directory, get_output_directory, get_available_letters
        import os
        
        from flask import current_app
        images_dir = get_sign_images_directory(current_app)
        output_dir = get_output_directory(current_app)
        available_letters = get_available_letters(current_app)
        
        return jsonify({
            'status': 'healthy',
            'service': 'speech-to-sign-whisper',
            'whisper_model_loaded': speech_processor.model_loaded,
            'images_directory': str(images_dir),
            'images_exist': os.path.exists(images_dir),
            'output_directory': str(output_dir),
            'output_directory_exists': os.path.exists(output_dir),
            'available_letters': available_letters,
            'letters_count': len(available_letters),
            'supported_audio_formats': list(ALLOWED_EXTENSIONS),
            'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),
            'processor_available': True,
            'sign_mappings_loaded': len(available_letters) > 0,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Error in health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'speech-to-sign-whisper',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@speech_to_sign_bp.route('/preview-audio', methods=['POST'])
def preview_audio_conversion():
    """Preview audio transcription without full processing"""
    try:
        if not speech_processor.model_loaded:
            return jsonify({'error': 'Whisper model not available'}), 503
        
        if 'audio' not in request.files:
            return jsonify({'error': 'Audio file is required'}), 400
        
        audio_file = request.files['audio']
        if not allowed_file(audio_file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
        # Save audio temporarily
        file_extension = secure_filename(audio_file.filename).rsplit('.', 1)[1].lower()
        temp_filename = f"{uuid.uuid4().hex}.{file_extension}"
        
        from flask import current_app
        upload_dir = current_app.config.get('UPLOAD_FOLDER', tempfile.gettempdir())
        os.makedirs(upload_dir, exist_ok=True)
        audio_path = os.path.join(upload_dir, temp_filename)
        
        try:
            audio_file.save(audio_path)
            
            # Get options from form
            data = request.form.to_dict()
            whisper_options = {
                'language': data.get('language'),
                'task': data.get('task', 'transcribe')
            }
            
            # Transcribe audio
            transcribed_text = speech_processor.transcribe_audio(audio_path, whisper_options)
            clean_text = clean_text_for_signs(transcribed_text)
            
            # Get available signs analysis
            from .text_to_sign import get_available_letters
            available_letters = get_available_letters(current_app)
            letters = list(clean_text.replace(' ', ''))
            available_signs = sum(1 for letter in letters if letter in available_letters)
            missing_signs = [letter for letter in set(letters) if letter not in available_letters]
            coverage_percentage = (available_signs / len(letters)) * 100 if letters else 0
            
            return jsonify({
                'success': True,
                'data': {
                    'transcribed_text': transcribed_text,
                    'clean_text': clean_text,
                    'letters': letters,
                    'total_letters': len(letters),
                    'available_signs': available_signs,
                    'missing_signs': missing_signs,
                    'coverage_percentage': round(coverage_percentage, 2),
                    'all_available': len(missing_signs) == 0,
                    'file_info': {
                        'filename': audio_file.filename,
                        'size_bytes': os.path.getsize(audio_path)
                    }
                }
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
    except Exception as e:
        logging.error(f"Error in preview_audio_conversion: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

# Legacy compatibility endpoints
@speech_to_sign_bp.route('/convert', methods=['POST'])
def convert_speech_to_sign():
    """Legacy endpoint - redirects to text conversion"""
    return convert_text_to_sign()

@speech_to_sign_bp.route('/process-text', methods=['POST'])
def process_text_for_signs():
    """Legacy endpoint - redirects to text conversion"""
    return convert_text_to_sign()
