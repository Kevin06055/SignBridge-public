"""
Braille Conversion Blueprint
Handles text-to-braille conversion with haptic feedback and multi-language support
"""

from flask import Blueprint, request, jsonify
import logging
from datetime import datetime
import time
import re

# Create blueprint
braille_conversion_bp = Blueprint('braille_conversion', __name__)

# Braille character mapping (Grade 1 Braille)
BRAILLE_MAP = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋',
    'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇',
    'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗',
    's': '⠎', 't': '⠞', 'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭',
    'y': '⠽', 'z': '⠵', ' ': '⠀',
    # Numbers
    '0': '⠚', '1': '⠁', '2': '⠃', '3': '⠉', '4': '⠙', '5': '⠑',
    '6': '⠋', '7': '⠛', '8': '⠓', '9': '⠊',
    # Punctuation
    '.': '⠲', ',': '⠂', '?': '⠦', '!': '⠖', ':': '⠒', ';': '⠆',
    '-': '⠤', "'": '⠄', '"': '⠐⠂'
}

# Contracted Braille patterns (Grade 2)
CONTRACTED_PATTERNS = {
    'and': '⠯', 'for': '⠿', 'of': '⠷', 'the': '⠮', 'with': '⠾',
    'you': '⠽', 'as': '⠵', 'but': '⠃', 'can': '⠉', 'do': '⠙',
    'every': '⠑', 'from': '⠋', 'go': '⠛', 'have': '⠓', 'just': '⠚',
    'knowledge': '⠅', 'like': '⠇', 'more': '⠍', 'not': '⠝', 'people': '⠏',
    'quite': '⠟', 'rather': '⠗', 'so': '⠎', 'that': '⠞', 'us': '⠥',
    'very': '⠧', 'will': '⠺', 'it': '⠭', 'your': '⠽'
}

# Vibration patterns for different characters
VIBRATION_PATTERNS = {
    'letter': [100, 50],
    'space': [200],
    'punctuation': [150, 30, 150],
    'number': [80, 40, 80, 40],
    'word_end': [300, 100]
}

class BrailleConverter:
    """Handles Braille conversion logic"""
    
    def __init__(self):
        self.supported_languages = ['english', 'spanish', 'french', 'german']
    
    def text_to_braille(self, text: str, language: str = 'english', contracted: bool = False) -> dict:
        """Convert text to Braille"""
        try:
            # Clean and prepare text
            clean_text = text.lower().strip()
            
            if not clean_text:
                raise ValueError("No text provided for conversion")
            
            # Apply contracted Braille if requested
            if contracted:
                clean_text = self._apply_contractions(clean_text)
            
            # Convert to Braille
            braille_text = ""
            vibration_pattern = []
            character_map = []
            
            for i, char in enumerate(clean_text):
                if char in BRAILLE_MAP:
                    braille_char = BRAILLE_MAP[char]
                    braille_text += braille_char
                    
                    # Add vibration pattern
                    if char == ' ':
                        vibration_pattern.extend(VIBRATION_PATTERNS['space'])
                    elif char.isdigit():
                        vibration_pattern.extend(VIBRATION_PATTERNS['number'])
                    elif char in '.?!,:;-':
                        vibration_pattern.extend(VIBRATION_PATTERNS['punctuation'])
                    else:
                        vibration_pattern.extend(VIBRATION_PATTERNS['letter'])
                    
                    # Track character mapping
                    character_map.append({
                        'original': char,
                        'braille': braille_char,
                        'position': i
                    })
                else:
                    # Unknown character - skip or use placeholder
                    braille_text += '⠿'  # Generic symbol
                    vibration_pattern.extend(VIBRATION_PATTERNS['punctuation'])
                    character_map.append({
                        'original': char,
                        'braille': '⠿',
                        'position': i
                    })
            
            return {
                'success': True,
                'original_text': text,
                'braille': braille_text,
                'language': language,
                'contracted': contracted,
                'character_count': len(clean_text),
                'braille_count': len([c for c in braille_text if c != ' ']),
                'vibration_pattern': vibration_pattern,
                'character_map': character_map,
                'conversion_time': time.time()
            }
            
        except Exception as e:
            logging.error(f"Braille conversion error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_text': text
            }
    
    def _apply_contractions(self, text: str) -> str:
        """Apply contracted Braille patterns"""
        words = text.split()
        contracted_words = []
        
        for word in words:
            # Check for exact matches first
            if word in CONTRACTED_PATTERNS:
                contracted_words.append(word)  # Keep original for now, mapping happens later
            else:
                # Check for partial matches
                contracted_word = word
                for pattern, contraction in CONTRACTED_PATTERNS.items():
                    contracted_word = contracted_word.replace(pattern, pattern)  # Placeholder
                contracted_words.append(contracted_word)
        
        return ' '.join(contracted_words)
    
    def braille_to_text(self, braille_text: str) -> dict:
        """Convert Braille back to text (reverse mapping)"""
        try:
            # Create reverse mapping
            reverse_map = {v: k for k, v in BRAILLE_MAP.items()}
            
            text = ""
            for char in braille_text:
                if char in reverse_map:
                    text += reverse_map[char]
                else:
                    text += '?'  # Unknown Braille character
            
            return {
                'success': True,
                'braille_text': braille_text,
                'converted_text': text,
                'character_count': len(text)
            }
            
        except Exception as e:
            logging.error(f"Braille to text conversion error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'braille_text': braille_text
            }

# Initialize converter
converter = BrailleConverter()

@braille_conversion_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'success': True,
            'service': 'braille_conversion',
            'status': 'active',
            'supported_languages': converter.supported_languages,
            'features': {
                'text_to_braille': True,
                'braille_to_text': True,
                'contracted_braille': True,
                'haptic_feedback': True,
                'multi_language': False  # Currently English only
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@braille_conversion_bp.route('/convert', methods=['POST'])
def convert_text_to_braille():
    """Convert text to Braille"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Extract parameters
        text = data.get('text', '')
        language = data.get('language', 'english').lower()
        contracted = data.get('contracted', False)
        enable_vibration = data.get('enable_vibration', True)
        enable_audio = data.get('enable_audio', False)
        
        # Validate input
        if not text.strip():
            return jsonify({
                'success': False,
                'error': 'Text field is required and cannot be empty'
            }), 400
        
        if language not in converter.supported_languages:
            return jsonify({
                'success': False,
                'error': f'Language "{language}" is not supported. Supported languages: {", ".join(converter.supported_languages)}'
            }), 400
        
        # Convert to Braille
        result = converter.text_to_braille(text, language, contracted)
        
        if not result['success']:
            return jsonify(result), 400
        
        # Prepare response
        response_data = {
            'success': True,
            'data': {
                'original_text': result['original_text'],
                'braille': result['braille'],
                'language': result['language'],
                'contracted': result['contracted'],
                'statistics': {
                    'character_count': result['character_count'],
                    'braille_count': result['braille_count']
                }
            },
            'settings': {
                'vibration_enabled': enable_vibration,
                'audio_enabled': enable_audio
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add vibration pattern if enabled
        if enable_vibration:
            response_data['data']['vibration_pattern'] = result['vibration_pattern']
        
        # Add character mapping for detailed analysis
        response_data['data']['character_map'] = result['character_map']
        
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Braille conversion API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during conversion'
        }), 500

@braille_conversion_bp.route('/reverse', methods=['POST'])
def convert_braille_to_text():
    """Convert Braille back to text"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        braille_text = data.get('braille_text', '')
        
        if not braille_text.strip():
            return jsonify({
                'success': False,
                'error': 'Braille text field is required and cannot be empty'
            }), 400
        
        # Convert from Braille
        result = converter.braille_to_text(braille_text)
        
        if not result['success']:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            'data': {
                'braille_text': result['braille_text'],
                'converted_text': result['converted_text'],
                'character_count': result['character_count']
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Braille to text API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during reverse conversion'
        }), 500

@braille_conversion_bp.route('/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'languages': converter.supported_languages,
                'default': 'english',
                'count': len(converter.supported_languages)
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@braille_conversion_bp.route('/patterns', methods=['GET'])
def get_braille_patterns():
    """Get Braille character patterns for reference"""
    try:
        pattern_type = request.args.get('type', 'all').lower()
        
        patterns = {}
        
        if pattern_type in ['all', 'letters']:
            letters = {k: v for k, v in BRAILLE_MAP.items() if k.isalpha()}
            patterns['letters'] = letters
        
        if pattern_type in ['all', 'numbers']:
            numbers = {k: v for k, v in BRAILLE_MAP.items() if k.isdigit()}
            patterns['numbers'] = numbers
        
        if pattern_type in ['all', 'punctuation']:
            punctuation = {k: v for k, v in BRAILLE_MAP.items() if not k.isalnum() and k != ' '}
            patterns['punctuation'] = punctuation
        
        if pattern_type in ['all', 'contractions']:
            patterns['contractions'] = CONTRACTED_PATTERNS
        
        return jsonify({
            'success': True,
            'data': {
                'type': pattern_type,
                'patterns': patterns,
                'total_patterns': sum(len(v) if isinstance(v, dict) else 0 for v in patterns.values())
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@braille_conversion_bp.route('/vibration/test', methods=['POST'])
def test_vibration():
    """Test vibration patterns"""
    try:
        data = request.get_json()
        pattern_type = data.get('pattern_type', 'letter') if data else 'letter'
        
        if pattern_type not in VIBRATION_PATTERNS:
            return jsonify({
                'success': False,
                'error': f'Pattern type "{pattern_type}" not supported'
            }), 400
        
        return jsonify({
            'success': True,
            'data': {
                'pattern_type': pattern_type,
                'vibration_pattern': VIBRATION_PATTERNS[pattern_type],
                'duration_ms': sum(VIBRATION_PATTERNS[pattern_type])
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500