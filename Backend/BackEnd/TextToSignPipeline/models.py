from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class OutputFormat(str, Enum):
    MP4 = "mp4"
    AVI = "avi"

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TextToSignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to convert to sign language")
    output_format: OutputFormat = Field(default=OutputFormat.MP4, description="Output video format")
    fps: float = Field(default=1.0, ge=0.5, le=10.0, description="Frames per second")
    frame_duration: float = Field(default=0.5, ge=0.1, le=3.0, description="Duration per letter in seconds")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()

class TextToSignResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str
    video_url: Optional[str] = None
    processing_details: Optional[Dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int = Field(ge=0, le=100)
    message: str
    start_time: str
    end_time: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    completed: bool = False

class QuizRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=1000, description="Text to generate quiz from")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="Quiz difficulty level")
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of questions to generate")

class QuizQuestion(BaseModel):
    question_id: str
    question: str
    options: List[str]
    correct_answer: str
    explanation: Optional[str] = None
    difficulty: DifficultyLevel

class QuizResponse(BaseModel):
    quiz_id: str
    questions: List[QuizQuestion]
    total_questions: int
    difficulty: DifficultyLevel
    expires_at: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class QuizSubmission(BaseModel):
    quiz_id: str
    answers: Dict[str, str]  # question_id -> selected_answer

class QuizResult(BaseModel):
    quiz_id: str
    score: int
    total_questions: int
    percentage: float
    passed: bool
    answers: List[Dict[str, Any]]
    completed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class AvailableSignsResponse(BaseModel):
    available_signs: List[str]
    total_count: int
    missing_signs: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    version: str
    available_signs: int
    uptime: Optional[str] = None

class ServiceInfo(BaseModel):
    service: str
    version: str
    status: str
    endpoints: List[str]
    description: str = "Convert text to sign language video output"

class ProcessingConfig(BaseModel):
    max_text_length: int = 1000
    default_fps: float = 1.0
    default_frame_duration: float = 0.5
    video_resolution: tuple = (640, 480)
    supported_formats: List[str] = ["mp4", "avi"]

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[Dict[str, Any]] = None