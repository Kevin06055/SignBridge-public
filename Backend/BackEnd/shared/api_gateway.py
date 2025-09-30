from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import structlog
from typing import Dict, Any
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings

logger = structlog.get_logger()

app = FastAPI(
    title="SignBridge API Gateway",
    description="Centralized API gateway for all SignBridge services",
    version=settings.version
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service endpoints mapping
SERVICES = {
    "sign_detection": f"http://{settings.sign_detection_host}:{settings.sign_detection_port}",
    "speech_to_sign": f"http://{settings.speech_to_sign_host}:{settings.speech_to_sign_port}",
    "text_summarization": f"http://{settings.text_summarization_host}:{settings.text_summarization_port}",
    "braille_conversion": f"http://{settings.braille_conversion_host}:{settings.braille_conversion_port}",
    "course_materials": f"http://{settings.course_materials_host}:{settings.course_materials_port}",
    "flask_api": "http://localhost:5000"  # Flask Sign Language API
}

@app.get("/")
async def root():
    """API Gateway health check"""
    return {
        "message": "SignBridge API Gateway",
        "version": settings.version,
        "services": list(SERVICES.keys())
    }

@app.get("/health")
async def health_check():
    """Check health of all services"""
    health_status = {}
    
    async with httpx.AsyncClient() as client:
        for service_name, service_url in SERVICES.items():
            try:
                if service_name == "flask_api":
                    response = await client.get(f"{service_url}/health", timeout=5.0)
                else:
                    response = await client.get(f"{service_url}/", timeout=5.0)
                
                health_status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "url": service_url,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            except Exception as e:
                health_status[service_name] = {
                    "status": "unhealthy",
                    "url": service_url,
                    "error": str(e)
                }
    
    return {
        "gateway_status": "healthy",
        "services": health_status
    }

# Proxy endpoints for each service
@app.api_route("/api/sign-detection/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_sign_detection(path: str, request):
    return await proxy_request("sign_detection", path, request)

@app.api_route("/api/speech-to-sign/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_speech_to_sign(path: str, request):
    return await proxy_request("speech_to_sign", path, request)

@app.api_route("/api/text-summarization/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_text_summarization(path: str, request):
    return await proxy_request("text_summarization", path, request)

@app.api_route("/api/braille-conversion/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_braille_conversion(path: str, request):
    return await proxy_request("braille_conversion", path, request)

@app.api_route("/api/course-materials/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_course_materials(path: str, request):
    return await proxy_request("course_materials", path, request)

@app.api_route("/api/sign-language/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_flask_api(path: str, request):
    return await proxy_request("flask_api", path, request)

async def proxy_request(service_name: str, path: str, request):
    """Proxy request to the appropriate service"""
    service_url = SERVICES.get(service_name)
    if not service_url:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    url = f"{service_url}/{path}"
    
    async with httpx.AsyncClient() as client:
        try:
            # Get request body and headers
            body = await request.body()
            headers = dict(request.headers)
            
            # Remove host header to avoid conflicts
            headers.pop("host", None)
            
            response = await client.request(
                method=request.method,
                url=url,
                content=body,
                headers=headers,
                params=request.query_params,
                timeout=30.0
            )
            
            return JSONResponse(
                status_code=response.status_code,
                content=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                headers=dict(response.headers)
            )
            
        except httpx.RequestError as e:
            logger.error("Proxy request failed", service=service_name, url=url, error=str(e))
            raise HTTPException(status_code=503, detail=f"Service {service_name} unavailable")
        except Exception as e:
            logger.error("Unexpected proxy error", service=service_name, error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "shared.api_gateway:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )