import uuid
import time
from typing import Any, Dict, List, Optional, Union
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from .api_models import (
    StandardAPIResponse, APIStatus, APIError, APIMetadata, 
    ErrorCode, ValidationError
)

logger = structlog.get_logger()

class APIResponseMiddleware(BaseHTTPMiddleware):
    """Middleware to standardize API responses and add metadata"""
    
    def __init__(self, app, service_name: str, service_version: str):
        super().__init__(app)
        self.service_name = service_name
        self.service_version = service_version
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to logs
        logger.bind(request_id=request_id)
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            processing_time = int((time.time() - start_time) * 1000)
            
            # Add metadata headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str(processing_time)
            response.headers["X-Service"] = self.service_name
            response.headers["X-Version"] = self.service_version
            
            return response
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            
            # Log the error
            logger.error("Request processing failed", 
                        error=str(e), 
                        request_id=request_id,
                        processing_time=processing_time)
            
            # Return standardized error response
            error_response = create_error_response(
                error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                message="Internal server error occurred",
                request_id=request_id,
                service=self.service_name,
                version=self.service_version,
                processing_time=processing_time
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.dict(),
                headers={
                    "X-Request-ID": request_id,
                    "X-Processing-Time": str(processing_time),
                    "X-Service": self.service_name,
                    "X-Version": self.service_version
                }
            )

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info("Request started",
                   method=request.method,
                   url=str(request.url),
                   headers=dict(request.headers),
                   request_id=getattr(request.state, 'request_id', None))
        
        response = await call_next(request)
        
        processing_time = time.time() - start_time
        
        # Log response
        logger.info("Request completed",
                   status_code=response.status_code,
                   processing_time=processing_time,
                   request_id=getattr(request.state, 'request_id', None))
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basic rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            ip: times for ip, times in self.requests.items() 
            if any(t > current_time - self.period for t in times)
        }
        
        # Check rate limit
        if client_ip in self.requests:
            self.requests[client_ip] = [
                t for t in self.requests[client_ip] 
                if t > current_time - self.period
            ]
            if len(self.requests[client_ip]) >= self.calls:
                error_response = create_error_response(
                    error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                    message=f"Rate limit exceeded: {self.calls} calls per {self.period} seconds",
                    request_id=getattr(request.state, 'request_id', None)
                )
                return JSONResponse(status_code=429, content=error_response.dict())
        else:
            self.requests[client_ip] = []
        
        self.requests[client_ip].append(current_time)
        
        return await call_next(request)

def create_success_response(
    data: Any = None,
    message: str = None,
    request_id: str = None,
    service: str = "unknown",
    version: str = "1.0.0",
    processing_time: int = None,
    user_id: str = None
) -> StandardAPIResponse:
    """Create a standardized success response"""
    
    return StandardAPIResponse(
        status=APIStatus.SUCCESS,
        message=message,
        data=data,
        metadata=APIMetadata(
            request_id=request_id or str(uuid.uuid4()),
            processing_time_ms=processing_time,
            service=service,
            version=version,
            user_id=user_id
        )
    )

def create_error_response(
    error_code: ErrorCode,
    message: str,
    details: Dict[str, Any] = None,
    request_id: str = None,
    service: str = "unknown",
    version: str = "1.0.0",
    processing_time: int = None,
    user_id: str = None
) -> StandardAPIResponse:
    """Create a standardized error response"""
    
    return StandardAPIResponse(
        status=APIStatus.ERROR,
        error=APIError(
            code=error_code,
            message=message,
            details=details,
            trace_id=request_id
        ),
        metadata=APIMetadata(
            request_id=request_id or str(uuid.uuid4()),
            processing_time_ms=processing_time,
            service=service,
            version=version,
            user_id=user_id
        )
    )

def create_validation_error_response(
    validation_errors: List[ValidationError],
    request_id: str = None,
    service: str = "unknown",
    version: str = "1.0.0"
) -> StandardAPIResponse:
    """Create a standardized validation error response"""
    
    return create_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Validation failed",
        details={"validation_errors": [error.dict() for error in validation_errors]},
        request_id=request_id,
        service=service,
        version=version
    )

def create_processing_response(
    message: str = "Processing request",
    data: Any = None,
    request_id: str = None,
    service: str = "unknown",
    version: str = "1.0.0"
) -> StandardAPIResponse:
    """Create a standardized processing response"""
    
    return StandardAPIResponse(
        status=APIStatus.PROCESSING,
        message=message,
        data=data,
        metadata=APIMetadata(
            request_id=request_id or str(uuid.uuid4()),
            service=service,
            version=version
        )
    )