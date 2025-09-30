"""
Course Material Pipeline Blueprint
Handles educational content management and access for sign language learning
"""

from flask import Blueprint, request, jsonify, send_file
import logging
from datetime import datetime, timedelta
import os
import json
import uuid
from pathlib import Path
import tempfile

# Create blueprint
course_material_bp = Blueprint('course_material', __name__)

# Sample course data structure
SAMPLE_COURSES = {
    'basic-asl': {
        'id': 'basic-asl',
        'title': 'Basic American Sign Language',
        'description': 'Introduction to fundamental ASL signs and grammar',
        'difficulty': 'beginner',
        'duration': '4 weeks',
        'lessons': 12,
        'category': 'sign_language',
        'instructor': 'Dr. Sarah Johnson',
        'rating': 4.8,
        'enrolled_count': 1250,
        'thumbnail': 'basic_asl_thumb.jpg',
        'created_at': '2024-01-15',
        'updated_at': '2024-09-01'
    },
    'intermediate-asl': {
        'id': 'intermediate-asl',
        'title': 'Intermediate ASL Conversation',
        'description': 'Build conversational skills and advanced vocabulary',
        'difficulty': 'intermediate',
        'duration': '6 weeks',
        'lessons': 18,
        'category': 'sign_language',
        'instructor': 'Mark Thompson',
        'rating': 4.7,
        'enrolled_count': 890,
        'thumbnail': 'intermediate_asl_thumb.jpg',
        'created_at': '2024-02-20',
        'updated_at': '2024-08-15'
    },
    'deaf-culture': {
        'id': 'deaf-culture',
        'title': 'Understanding Deaf Culture',
        'description': 'Learn about Deaf community history, values, and traditions',
        'difficulty': 'beginner',
        'duration': '3 weeks',
        'lessons': 8,
        'category': 'cultural',
        'instructor': 'Lisa Williams',
        'rating': 4.9,
        'enrolled_count': 670,
        'thumbnail': 'deaf_culture_thumb.jpg',
        'created_at': '2024-03-10',
        'updated_at': '2024-09-05'
    },
    'fingerspelling-mastery': {
        'id': 'fingerspelling-mastery',
        'title': 'Fingerspelling Mastery',
        'description': 'Master the American Sign Language alphabet and number system',
        'difficulty': 'beginner',
        'duration': '2 weeks',
        'lessons': 6,
        'category': 'fundamentals',
        'instructor': 'Jennifer Davis',
        'rating': 4.6,
        'enrolled_count': 2100,
        'thumbnail': 'fingerspelling_thumb.jpg',
        'created_at': '2024-01-05',
        'updated_at': '2024-08-20'
    },
    'sign-interpretation': {
        'id': 'sign-interpretation',
        'title': 'Professional Sign Language Interpretation',
        'description': 'Advanced course for aspiring professional interpreters',
        'difficulty': 'advanced',
        'duration': '12 weeks',
        'lessons': 36,
        'category': 'professional',
        'instructor': 'Robert Martinez',
        'rating': 4.8,
        'enrolled_count': 320,
        'thumbnail': 'interpretation_thumb.jpg',
        'created_at': '2024-04-01',
        'updated_at': '2024-09-10'
    }
}

SAMPLE_LESSONS = {
    'basic-asl': [
        {'id': 'lesson-1', 'title': 'Introduction to ASL', 'duration': '15 min', 'type': 'video'},
        {'id': 'lesson-2', 'title': 'Basic Greetings', 'duration': '20 min', 'type': 'video'},
        {'id': 'lesson-3', 'title': 'Family Signs', 'duration': '25 min', 'type': 'interactive'},
        {'id': 'lesson-4', 'title': 'Practice Session 1', 'duration': '10 min', 'type': 'practice'},
    ],
    'fingerspelling-mastery': [
        {'id': 'fs-1', 'title': 'A-F Letters', 'duration': '12 min', 'type': 'video'},
        {'id': 'fs-2', 'title': 'G-L Letters', 'duration': '12 min', 'type': 'video'},
        {'id': 'fs-3', 'title': 'M-R Letters', 'duration': '12 min', 'type': 'video'},
        {'id': 'fs-4', 'title': 'S-Z Letters', 'duration': '12 min', 'type': 'video'},
        {'id': 'fs-5', 'title': 'Numbers 0-9', 'duration': '15 min', 'type': 'video'},
        {'id': 'fs-6', 'title': 'Speed Practice', 'duration': '20 min', 'type': 'practice'},
    ]
}

CATEGORIES = [
    {'id': 'sign_language', 'name': 'Sign Language', 'description': 'Core ASL learning courses'},
    {'id': 'cultural', 'name': 'Deaf Culture', 'description': 'Understanding Deaf community and culture'},
    {'id': 'fundamentals', 'name': 'Fundamentals', 'description': 'Basic skills and alphabet'},
    {'id': 'professional', 'name': 'Professional', 'description': 'Career and interpretation skills'},
    {'id': 'practice', 'name': 'Practice', 'description': 'Skill reinforcement and drills'}
]

class CourseManager:
    """Manages course materials and user progress"""
    
    def __init__(self):
        self.courses = SAMPLE_COURSES.copy()
        self.lessons = SAMPLE_LESSONS.copy()
        self.categories = CATEGORIES.copy()
        self.user_progress = {}  # In production, this would be database-backed
    
    def get_all_courses(self, category=None, difficulty=None, search=None):
        """Get filtered list of courses"""
        courses = list(self.courses.values())
        
        # Apply filters
        if category:
            courses = [c for c in courses if c['category'] == category]
        
        if difficulty:
            courses = [c for c in courses if c['difficulty'] == difficulty]
        
        if search:
            search_lower = search.lower()
            courses = [c for c in courses if 
                      search_lower in c['title'].lower() or 
                      search_lower in c['description'].lower() or
                      search_lower in c['instructor'].lower()]
        
        # Sort by rating and enrollment
        courses.sort(key=lambda x: (x['rating'], x['enrolled_count']), reverse=True)
        
        return courses
    
    def get_course_details(self, course_id):
        """Get detailed course information"""
        if course_id not in self.courses:
            return None
        
        course = self.courses[course_id].copy()
        course['lessons_list'] = self.lessons.get(course_id, [])
        
        return course
    
    def enroll_user(self, user_id, course_id):
        """Enroll user in a course"""
        if course_id not in self.courses:
            return False, "Course not found"
        
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}
        
        if course_id not in self.user_progress[user_id]:
            self.user_progress[user_id][course_id] = {
                'enrolled_at': datetime.now().isoformat(),
                'progress': 0,
                'completed_lessons': [],
                'current_lesson': 0,
                'quiz_scores': {},
                'last_accessed': datetime.now().isoformat()
            }
            
            # Increment enrollment count
            self.courses[course_id]['enrolled_count'] += 1
            
            return True, "Successfully enrolled"
        else:
            return False, "Already enrolled in this course"
    
    def update_progress(self, user_id, course_id, lesson_id, completed=True, score=None):
        """Update user progress for a lesson"""
        if (user_id not in self.user_progress or 
            course_id not in self.user_progress[user_id]):
            return False, "User not enrolled in course"
        
        user_course = self.user_progress[user_id][course_id]
        
        if completed and lesson_id not in user_course['completed_lessons']:
            user_course['completed_lessons'].append(lesson_id)
        
        if score is not None:
            user_course['quiz_scores'][lesson_id] = score
        
        # Update progress percentage
        total_lessons = len(self.lessons.get(course_id, []))
        if total_lessons > 0:
            user_course['progress'] = len(user_course['completed_lessons']) / total_lessons * 100
        
        user_course['last_accessed'] = datetime.now().isoformat()
        
        return True, "Progress updated successfully"
    
    def get_user_courses(self, user_id):
        """Get courses user is enrolled in"""
        if user_id not in self.user_progress:
            return []
        
        enrolled_courses = []
        for course_id, progress in self.user_progress[user_id].items():
            course = self.courses[course_id].copy()
            course.update({
                'user_progress': progress['progress'],
                'completed_lessons': len(progress['completed_lessons']),
                'last_accessed': progress['last_accessed'],
                'enrolled_at': progress['enrolled_at']
            })
            enrolled_courses.append(course)
        
        return enrolled_courses

# Initialize course manager
course_manager = CourseManager()

@course_material_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'success': True,
            'service': 'course_material',
            'status': 'active',
            'statistics': {
                'total_courses': len(course_manager.courses),
                'total_categories': len(course_manager.categories),
                'total_enrolled_users': len(course_manager.user_progress)
            },
            'features': {
                'course_browsing': True,
                'user_enrollment': True,
                'progress_tracking': True,
                'categories': True,
                'search_filter': True
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@course_material_bp.route('/courses', methods=['GET'])
def get_courses():
    """Get list of available courses with optional filtering"""
    try:
        # Get query parameters
        category = request.args.get('category')
        difficulty = request.args.get('difficulty')
        search = request.args.get('search')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        # Get filtered courses
        courses = course_manager.get_all_courses(category, difficulty, search)
        
        # Implement pagination
        total_courses = len(courses)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_courses = courses[start_idx:end_idx]
        
        return jsonify({
            'success': True,
            'data': {
                'courses': paginated_courses,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total_courses,
                    'pages': (total_courses + per_page - 1) // per_page
                },
                'filters_applied': {
                    'category': category,
                    'difficulty': difficulty,
                    'search': search
                }
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Course listing error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve courses'
        }), 500

@course_material_bp.route('/courses/<course_id>', methods=['GET'])
def get_course_details(course_id):
    """Get detailed information about a specific course"""
    try:
        course = course_manager.get_course_details(course_id)
        
        if not course:
            return jsonify({
                'success': False,
                'error': 'Course not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': {
                'course': course
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Course details error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve course details'
        }), 500

@course_material_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get list of course categories"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'categories': course_manager.categories
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@course_material_bp.route('/enroll', methods=['POST'])
def enroll_in_course():
    """Enroll a user in a course"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        user_id = data.get('user_id')
        course_id = data.get('course_id')
        
        if not user_id or not course_id:
            return jsonify({
                'success': False,
                'error': 'user_id and course_id are required'
            }), 400
        
        success, message = course_manager.enroll_user(user_id, course_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'data': {
                    'user_id': user_id,
                    'course_id': course_id,
                    'enrolled_at': datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
            
    except Exception as e:
        logging.error(f"Enrollment error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to enroll in course'
        }), 500

@course_material_bp.route('/progress', methods=['POST'])
def update_progress():
    """Update user progress for a lesson"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        user_id = data.get('user_id')
        course_id = data.get('course_id')
        lesson_id = data.get('lesson_id')
        completed = data.get('completed', True)
        score = data.get('score')
        
        if not all([user_id, course_id, lesson_id]):
            return jsonify({
                'success': False,
                'error': 'user_id, course_id, and lesson_id are required'
            }), 400
        
        success, message = course_manager.update_progress(
            user_id, course_id, lesson_id, completed, score
        )
        
        if success:
            user_course = course_manager.user_progress[user_id][course_id]
            return jsonify({
                'success': True,
                'message': message,
                'data': {
                    'progress_percentage': user_course['progress'],
                    'completed_lessons': len(user_course['completed_lessons']),
                    'lesson_id': lesson_id,
                    'completed': completed
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
            
    except Exception as e:
        logging.error(f"Progress update error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to update progress'
        }), 500

@course_material_bp.route('/my-courses/<user_id>', methods=['GET'])
def get_user_courses(user_id):
    """Get courses that a user is enrolled in"""
    try:
        courses = course_manager.get_user_courses(user_id)
        
        return jsonify({
            'success': True,
            'data': {
                'user_id': user_id,
                'enrolled_courses': courses,
                'total_enrolled': len(courses)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"User courses error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve user courses'
        }), 500

@course_material_bp.route('/search', methods=['POST'])
def search_courses():
    """Advanced course search"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        query = data.get('query', '')
        filters = data.get('filters', {})
        
        courses = course_manager.get_all_courses(
            category=filters.get('category'),
            difficulty=filters.get('difficulty'),
            search=query
        )
        
        return jsonify({
            'success': True,
            'data': {
                'query': query,
                'filters': filters,
                'results': courses,
                'total_results': len(courses)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Course search error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to search courses'
        }), 500

@course_material_bp.route('/stats', methods=['GET'])
def get_course_statistics():
    """Get course platform statistics"""
    try:
        total_courses = len(course_manager.courses)
        total_lessons = sum(len(lessons) for lessons in course_manager.lessons.values())
        total_users = len(course_manager.user_progress)
        
        # Calculate difficulty distribution
        difficulty_stats = {}
        for course in course_manager.courses.values():
            difficulty = course['difficulty']
            difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
        
        # Calculate category distribution
        category_stats = {}
        for course in course_manager.courses.values():
            category = course['category']
            category_stats[category] = category_stats.get(category, 0) + 1
        
        return jsonify({
            'success': True,
            'data': {
                'overview': {
                    'total_courses': total_courses,
                    'total_lessons': total_lessons,
                    'total_users': total_users,
                    'average_rating': sum(c['rating'] for c in course_manager.courses.values()) / total_courses if total_courses > 0 else 0
                },
                'distributions': {
                    'by_difficulty': difficulty_stats,
                    'by_category': category_stats
                },
                'popular_courses': sorted(
                    course_manager.courses.values(),
                    key=lambda x: x['enrolled_count'],
                    reverse=True
                )[:5]
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Course statistics error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve statistics'
        }), 500