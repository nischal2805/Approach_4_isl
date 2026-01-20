"""
ISL Backend - API Routes for Sign-to-Text
"""
import io
import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel

from services.sign_to_text import get_sign_to_text_service
from services.llm_service import get_llm_service

router = APIRouter(prefix="/sign-to-text", tags=["Sign to Text"])


class SignPrediction(BaseModel):
    sign: str
    confidence: float
    start_frame: int = 0
    end_frame: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    top_predictions: dict = {}


class SignToTextResponse(BaseModel):
    success: bool
    signs: List[SignPrediction]
    sentence: str
    message: str = ""


class AvailableClassesResponse(BaseModel):
    count: int
    classes: List[str]


@router.get("/classes", response_model=AvailableClassesResponse)
async def get_available_classes():
    """Get list of all recognizable sign classes."""
    service = get_sign_to_text_service()
    classes = service.class_names
    return AvailableClassesResponse(count=len(classes), classes=classes)


@router.post("/test-upload")
async def test_upload(video: UploadFile = File(...)):
    """Simple test endpoint to verify uploads work."""
    import logging
    logging.info(f"TEST UPLOAD: filename={video.filename}, content_type={video.content_type}")
    try:
        content = await video.read()
        logging.info(f"TEST UPLOAD: Read {len(content)} bytes")
        return {"success": True, "filename": video.filename, "size": len(content)}
    except Exception as e:
        logging.error(f"TEST UPLOAD ERROR: {e}")
        return {"success": False, "error": str(e)}


@router.post("/video", response_model=SignToTextResponse)
async def process_video(video: UploadFile = File(...)):
    """
    Process a video file and return detected signs.
    
    Accepts: MP4, AVI, MOV, WebM video files
    Returns: List of detected signs with confidence and timestamps
    """
    # Log what we received for debugging
    import logging
    import traceback
    logging.info(f"=== VIDEO UPLOAD ===")
    logging.info(f"Filename: {video.filename}")
    logging.info(f"Content-Type: {video.content_type}")
    logging.info(f"Size: {video.size if hasattr(video, 'size') else 'unknown'}")
    
    # Accept any video type - Android might send different content types
    # Don't be strict about content type validation
    
    try:
        service = get_sign_to_text_service()
        llm = get_llm_service()
        
        # Read video bytes
        video_bytes = await video.read()
        
        # Process video
        results = service.process_video_bytes(video_bytes)
        
        # Convert to response model
        signs = [SignPrediction(**r) for r in results]
        
        # Create gloss from detected signs
        gloss = " ".join([s.sign for s in signs])
        
        # Use LLM to correct grammar (if available)
        sentence = await llm.correct_grammar(gloss)
        
        return SignToTextResponse(
            success=True,
            signs=signs,
            sentence=sentence,
            message=f"Detected {len(signs)} signs"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/frames", response_model=SignToTextResponse)
async def process_frames(
    frames: List[UploadFile] = File(...),
):
    """
    Process a batch of frame images and return prediction.
    
    Expects 30 frames (1 second of video) as JPEG/PNG images.
    """
    if len(frames) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 10 frames, got {len(frames)}"
        )
    
    try:
        service = get_sign_to_text_service()
        
        # Convert uploaded images to numpy arrays
        np_frames = []
        for frame_file in frames:
            contents = await frame_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                np_frames.append(img)
        
        if len(np_frames) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Could only decode {len(np_frames)} valid frames"
            )
        
        # Process frames
        result = service.process_frames(np_frames)
        
        # Wrap in list for consistency
        signs = [SignPrediction(
            sign=result["sign"],
            confidence=result["confidence"],
            end_frame=result["num_frames"],
            top_predictions=result["top_predictions"]
        )]
        
        return SignToTextResponse(
            success=True,
            signs=signs,
            sentence=result["sign"],
            message=f"Processed {len(np_frames)} frames"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/frame-bytes")
async def process_frame_bytes(
    width: int = Form(...),
    height: int = Form(...),
    frames_data: bytes = File(...)
):
    """
    Process raw frame bytes (for streaming from mobile app).
    
    Expects concatenated raw BGR bytes for multiple frames.
    """
    try:
        service = get_sign_to_text_service()
        
        frame_size = width * height * 3  # BGR
        num_frames = len(frames_data) // frame_size
        
        if num_frames < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 10 frames, got {num_frames}"
            )
        
        # Decode frames
        np_frames = []
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame_bytes = frames_data[start:end]
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
            np_frames.append(frame)
        
        # Process
        result = service.process_frames(np_frames)
        
        return {
            "success": True,
            "sign": result["sign"],
            "confidence": result["confidence"],
            "num_frames": num_frames,
            "top_predictions": result["top_predictions"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
