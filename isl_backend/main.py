"""
ISL Backend - FastAPI Application

Main entry point for the ISL Translation API.
Provides:
- Sign-to-Text: Convert ISL sign videos to English text
- Text-to-Sign: Convert English text to ISL sign animations
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import HOST, PORT, FRAMES_WORD_LEVEL
from routes.sign_to_text import router as sign_to_text_router
from routes.text_to_sign import router as text_to_sign_router
from services.sign_to_text import get_sign_to_text_service
from services.text_to_sign import get_text_to_sign_service
from services.llm_service import get_llm_service

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # More verbose for debugging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load models on startup."""
    logger.info("=" * 50)
    logger.info("ISL Backend Starting...")
    logger.info("=" * 50)
    
    # Load Sign-to-Text model
    logger.info("Loading Sign-to-Text service...")
    s2t = get_sign_to_text_service()
    if s2t.is_loaded:
        logger.info(f"✓ Sign-to-Text ready ({len(s2t.class_names)} classes)")
    else:
        logger.error("✗ Sign-to-Text failed to load!")
    
    # Load Text-to-Sign mappings
    logger.info("Loading Text-to-Sign service...")
    t2s = get_text_to_sign_service()
    if t2s.is_loaded:
        logger.info(f"✓ Text-to-Sign ready ({len(t2s.available_words)} words)")
    else:
        logger.error("✗ Text-to-Sign failed to load!")
    
    # Check LLM availability
    logger.info("Checking LLM (Ollama) availability...")
    llm = get_llm_service()
    if await llm.check_availability():
        logger.info("✓ LLM ready (Ollama)")
    else:
        logger.warning("⚠ LLM not available - grammar correction disabled")
        logger.warning("  Run: ollama pull llama3.2:1b")
    
    logger.info("=" * 50)
    logger.info(f"Server running at http://{HOST}:{PORT}")
    logger.info("API docs at http://localhost:8000/docs")
    logger.info("=" * 50)
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    s2t.cleanup()


# Create FastAPI app
app = FastAPI(
    title="ISL Translation API",
    description="""
    Indian Sign Language Translation API
    
    ## Features
    
    ### Sign-to-Text
    - Upload video files to detect ISL signs
    - Streaming frame processing for real-time apps
    - Returns detected signs with confidence scores
    
    ### Text-to-Sign  
    - Convert English sentences to ISL animations
    - Get frame sequences for each word
    - Supports 100+ common ISL words
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handlers for debugging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all exceptions and log them."""
    logger.error(f"Global exception: {type(exc).__name__}: {exc}")
    logger.error(f"Request: {request.method} {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {str(exc)}"}
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Log HTTP exceptions."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    logger.warning(f"Request: {request.method} {request.url}")
    logger.warning(f"Headers: {dict(request.headers)}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# Mount static files for sign frames
if FRAMES_WORD_LEVEL.exists():
    app.mount("/static/frames", StaticFiles(directory=str(FRAMES_WORD_LEVEL)), name="frames")

# Include routers
app.include_router(sign_to_text_router)
app.include_router(text_to_sign_router)


@app.get("/", tags=["Health"])
async def root():
    """API health check."""
    return {
        "status": "running",
        "api": "ISL Translation API",
        "version": "1.0.0",
        "endpoints": {
            "sign_to_text": "/sign-to-text",
            "text_to_sign": "/text-to-sign",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    s2t = get_sign_to_text_service()
    t2s = get_text_to_sign_service()
    llm = get_llm_service()
    
    return {
        "status": "healthy" if s2t.is_loaded and t2s.is_loaded else "degraded",
        "services": {
            "sign_to_text": {
                "loaded": s2t.is_loaded,
                "classes_count": len(s2t.class_names) if s2t.is_loaded else 0
            },
            "text_to_sign": {
                "loaded": t2s.is_loaded,
                "words_count": len(t2s.available_words) if t2s.is_loaded else 0
            },
            "llm": {
                "available": llm.is_available,
                "model": "llama3.2:1b"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
