import httpx
import asyncio
from typing import Any, Dict, List, Optional, Union
from fastapi import HTTPException
import structlog

from .api_models import StandardAPIResponse, APIStatus, ErrorCode
from .config import settings

logger = structlog.get_logger()

class APIClient:
    """Standardized API client for inter-service communication"""
    
    def __init__(self, base_url: str, api_key: str = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or settings.api_keys[0] if settings.api_keys else None
        self.timeout = timeout
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers=self._get_default_headers()
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    def _get_default_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"SignBridge-API-Client/{settings.version}"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def get(
        self, 
        endpoint: str, 
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> StandardAPIResponse:
        """Make GET request"""
        return await self._make_request("GET", endpoint, params=params, headers=headers)
    
    async def post(
        self, 
        endpoint: str, 
        data: Any = None,
        json: Dict[str, Any] = None,
        files: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> StandardAPIResponse:
        """Make POST request"""
        return await self._make_request(
            "POST", endpoint, data=data, json=json, files=files, headers=headers
        )
    
    async def put(
        self, 
        endpoint: str, 
        json: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> StandardAPIResponse:
        """Make PUT request"""
        return await self._make_request("PUT", endpoint, json=json, headers=headers)
    
    async def delete(
        self, 
        endpoint: str, 
        headers: Dict[str, str] = None
    ) -> StandardAPIResponse:
        """Make DELETE request"""
        return await self._make_request("DELETE", endpoint, headers=headers)
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> StandardAPIResponse:
        """Make HTTP request with error handling and retry logic"""
        
        if not self.client:
            raise RuntimeError("APIClient must be used as async context manager")
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Merge headers
        request_headers = self._get_default_headers()
        if kwargs.get('headers'):
            request_headers.update(kwargs.pop('headers'))
        
        kwargs['headers'] = request_headers
        
        logger.info("Making API request", method=method, url=url)
        
        try:
            response = await self.client.request(method, url, **kwargs)
            
            # Handle different response types
            if response.headers.get('content-type', '').startswith('application/json'):
                response_data = response.json()
            else:
                response_data = {"raw_response": response.text}
            
            # Check if response follows our standard format
            if isinstance(response_data, dict) and 'status' in response_data:
                api_response = StandardAPIResponse(**response_data)
            else:
                # Wrap non-standard responses
                api_response = StandardAPIResponse(
                    status=APIStatus.SUCCESS if response.is_success else APIStatus.ERROR,
                    data=response_data,
                    metadata={
                        "request_id": response.headers.get('X-Request-ID'),
                        "service": response.headers.get('X-Service', 'unknown'),
                        "version": response.headers.get('X-Version', '1.0.0'),
                        "processing_time_ms": response.headers.get('X-Processing-Time')
                    }
                )
            
            # Handle HTTP errors
            if not response.is_success:
                error_msg = f"HTTP {response.status_code}: {response.reason_phrase}"
                if api_response.error:
                    error_msg = api_response.error.message
                
                raise HTTPException(status_code=response.status_code, detail=error_msg)
            
            logger.info("API request successful", 
                       method=method, url=url, 
                       status_code=response.status_code)
            
            return api_response
            
        except httpx.TimeoutException:
            logger.error("API request timeout", method=method, url=url)
            raise HTTPException(
                status_code=408, 
                detail="Request timeout - service may be unavailable"
            )
        
        except httpx.ConnectError:
            logger.error("API connection error", method=method, url=url)
            raise HTTPException(
                status_code=503, 
                detail="Service unavailable - unable to connect"
            )
        
        except Exception as e:
            logger.error("API request failed", method=method, url=url, error=str(e))
            raise HTTPException(
                status_code=500, 
                detail=f"API request failed: {str(e)}"
            )

class ServiceRegistry:
    """Registry for managing service endpoints"""
    
    def __init__(self):
        self.services = {
            "sign-detection": {
                "url": f"http://{settings.sign_detection_host}:{settings.sign_detection_port}",
                "health_endpoint": "/health"
            },
            "speech-to-sign": {
                "url": f"http://{settings.speech_to_sign_host}:{settings.speech_to_sign_port}",
                "health_endpoint": "/health"
            },
            "text-summarization": {
                "url": "http://localhost:8003",  # Default port for text summarization
                "health_endpoint": "/health"
            },
            "braille-conversion": {
                "url": "http://localhost:8004",  # Default port for Braille conversion
                "health_endpoint": "/health"
            },
            "course-materials": {
                "url": "http://localhost:8005",  # Default port for course materials
                "health_endpoint": "/health"
            }
        }
    
    def get_service_url(self, service_name: str) -> str:
        """Get service URL by name"""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        return self.services[service_name]["url"]
    
    def get_health_endpoint(self, service_name: str) -> str:
        """Get health check endpoint for service"""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        return self.services[service_name]["health_endpoint"]
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        try:
            service_url = self.get_service_url(service_name)
            health_endpoint = self.get_health_endpoint(service_name)
            
            async with APIClient(service_url) as client:
                response = await client.get(health_endpoint)
                return response.status == APIStatus.SUCCESS
                
        except Exception as e:
            logger.error(f"Health check failed for {service_name}", error=str(e))
            return False
    
    async def check_all_services_health(self) -> Dict[str, bool]:
        """Check health of all registered services"""
        health_results = {}
        
        tasks = [
            (service_name, self.check_service_health(service_name))
            for service_name in self.services.keys()
        ]
        
        for service_name, health_task in tasks:
            try:
                is_healthy = await health_task
                health_results[service_name] = is_healthy
            except Exception:
                health_results[service_name] = False
        
        return health_results

# Global service registry instance
service_registry = ServiceRegistry()

# Convenience functions for common operations
async def call_sign_detection_service(endpoint: str, **kwargs) -> StandardAPIResponse:
    """Call sign detection service"""
    service_url = service_registry.get_service_url("sign-detection")
    async with APIClient(service_url) as client:
        return await client.post(endpoint, **kwargs)

async def call_speech_to_sign_service(endpoint: str, **kwargs) -> StandardAPIResponse:
    """Call speech-to-sign service"""
    service_url = service_registry.get_service_url("speech-to-sign")
    async with APIClient(service_url) as client:
        return await client.post(endpoint, **kwargs)

async def call_text_summarization_service(endpoint: str, **kwargs) -> StandardAPIResponse:
    """Call text summarization service"""
    service_url = service_registry.get_service_url("text-summarization")
    async with APIClient(service_url) as client:
        return await client.post(endpoint, **kwargs)

async def call_braille_conversion_service(endpoint: str, **kwargs) -> StandardAPIResponse:
    """Call Braille conversion service"""
    service_url = service_registry.get_service_url("braille-conversion")
    async with APIClient(service_url) as client:
        return await client.post(endpoint, **kwargs)

async def call_course_materials_service(endpoint: str, **kwargs) -> StandardAPIResponse:
    """Call course materials service"""
    service_url = service_registry.get_service_url("course-materials")
    async with APIClient(service_url) as client:
        return await client.post(endpoint, **kwargs)