#!/usr/bin/env python3
"""
ISL Translation - FastAPI Inference Server

Deployable to AWS Lambda via Mangum or run locally.
"""

import os
import io
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ISL Translation API",
    description="Indian Sign Language to English Translation",
    version="1.0.0"
)

# CORS for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
translator = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class TranslationResponse(BaseModel):
    translation: str
    confidence: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


def load_model():
    """Load the translation model."""
    global translator
    
    if translator is not None:
        return translator
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models import ISLTranslator
        
        # Load model
        checkpoint_path = os.environ.get("MODEL_PATH", "checkpoints/best_model.pt")
        
        translator = ISLTranslator(
            t5_model_name='t5-small',
            freeze_i3d=True,
            lstm_hidden=512,
            lstm_layers=2
        )
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            translator.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, using random weights")
        
        translator = translator.to(device)
        translator.eval()
        
        logger.info(f"Model loaded on {device}")
        return translator
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_video(video_bytes: bytes, num_frames: int = 30, 
                     frame_size: int = 224) -> torch.Tensor:
    """Preprocess video bytes to tensor."""
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name
    
    try:
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise HTTPException(status_code=400, detail="Empty video")
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((frame_size, frame_size, 3), dtype=np.uint8))
                continue
            
            frame = cv2.resize(frame, (frame_size, frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Stack and normalize
        video = np.stack(frames, axis=0)  # [T, H, W, 3]
        video = video.transpose(3, 0, 1, 2)  # [3, T, H, W]
        video = video.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
        video = (video - mean) / std
        
        # Add batch dimension
        video = torch.from_numpy(video).float().unsqueeze(0)
        
        return video
        
    finally:
        os.unlink(temp_path)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=translator is not None,
        device=device
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate_video(video: UploadFile = File(...)):
    """
    Translate sign language video to English text.
    
    Upload a video file (mp4, mov, avi) containing sign language.
    Returns the English translation.
    """
    import time
    
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    start_time = time.time()
    
    try:
        # Read video bytes
        video_bytes = await video.read()
        
        # Preprocess
        video_tensor = preprocess_video(video_bytes)
        video_tensor = video_tensor.to(device)
        
        # Translate
        with torch.no_grad():
            translations = translator.translate(
                video_tensor, 
                max_length=50, 
                num_beams=4
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return TranslationResponse(
            translation=translations[0],
            confidence=0.85,  # TODO: compute actual confidence
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate_base64", response_model=TranslationResponse)
async def translate_base64(data: dict):
    """
    Translate base64-encoded video.
    
    Useful for mobile apps that send video as base64 string.
    """
    import time
    
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if "video" not in data:
        raise HTTPException(status_code=400, detail="Missing 'video' field")
    
    start_time = time.time()
    
    try:
        # Decode base64
        video_bytes = base64.b64decode(data["video"])
        
        # Preprocess
        video_tensor = preprocess_video(video_bytes)
        video_tensor = video_tensor.to(device)
        
        # Translate
        with torch.no_grad():
            translations = translator.translate(
                video_tensor, 
                max_length=50, 
                num_beams=4
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return TranslationResponse(
            translation=translations[0],
            confidence=0.85,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# AWS Lambda handler (via Mangum)
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    handler = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
