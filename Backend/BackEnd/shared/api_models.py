from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

class APIStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PROCESSING = "processing"

class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    INVALID_FILE_FORMAT = "INVALID_FILE_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    PROCESSING_ERROR = "PROCESSING_ERROR"

class APIError(BaseModel):
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.utcnow()
    trace_id: Optional[str] = None

class APIMetadata(BaseModel):
    request_id: str
    timestamp: datetime = datetime.utcnow()
    processing_time_ms: Optional[int] = None
    service: str
    version: str
    user_id: Optional[str] = None

class StandardAPIResponse(BaseModel):
    """Standardized API response format for all services"""
    status: APIStatus
    message: Optional[str] = None
    data: Optional[Union[Dict[str, Any], List[Any], Any]] = None
    error: Optional[APIError] = None
    metadata: APIMetadata

class PaginationRequest(BaseModel):
    page: int = 1
    limit: int = 20
    sort_by: Optional[str] = None
    sort_order: Optional[str] = "asc"

class PaginationResponse(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    has_next: bool
    has_prev: bool

class HealthCheckResponse(BaseModel):
    service: str
    status: str
    version: str
    timestamp: datetime = datetime.utcnow()
    uptime: Optional[int] = None
    dependencies: Optional[Dict[str, str]] = None
    performance: Optional[Dict[str, float]] = None

class FileUploadRequest(BaseModel):
    max_size_mb: int = 50
    allowed_formats: List[str] = ["jpg", "jpeg", "png", "pdf", "txt", "docx"]
    preserve_metadata: bool = True

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    content_type: str
    upload_url: Optional[str] = None
    processing_status: str = "uploaded"
    metadata: Optional[Dict[str, Any]] = None

class ValidationError(BaseModel):
    field: str
    message: str
    code: str