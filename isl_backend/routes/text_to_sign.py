"""
ISL Backend - API Routes for Text-to-Sign
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from services.text_to_sign import get_text_to_sign_service
from config import FRAMES_WORD_LEVEL

router = APIRouter(prefix="/text-to-sign", tags=["Text to Sign"])


class WordResult(BaseModel):
    word: str
    found: bool
    frames_count: int
    frame_paths: List[str]


class TextToSignResponse(BaseModel):
    success: bool
    original_text: str
    words: List[WordResult]
    found_count: int
    total_count: int


class AvailableWordsResponse(BaseModel):
    count: int
    words: List[str]


class FrameData(BaseModel):
    filename: str
    data_base64: str


class AnimationWord(BaseModel):
    word: str
    found: bool
    frames: List[FrameData]


class AnimationResponse(BaseModel):
    success: bool
    original_text: str
    animations: List[AnimationWord]


@router.get("/words", response_model=AvailableWordsResponse)
async def get_available_words():
    """Get list of all available sign words."""
    service = get_text_to_sign_service()
    words = service.get_available_words()
    return AvailableWordsResponse(count=len(words), words=words)


@router.get("/translate", response_model=TextToSignResponse)
async def translate_text(text: str = Query(..., description="English text to translate")):
    """
    Translate English text to ISL signs.
    
    Returns paths to frame images for each word that has a sign available.
    Use /frames/{path} endpoint to get individual frame images.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        service = get_text_to_sign_service()
        result = service.text_to_signs(text)
        
        words = [WordResult(**w) for w in result["words"]]
        
        return TextToSignResponse(
            success=True,
            original_text=result["original_text"],
            words=words,
            found_count=result["found_count"],
            total_count=result["total_count"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/animation", response_model=AnimationResponse)
async def get_animation_data(text: str = Query(..., description="English text to animate")):
    """
    Get complete animation data with base64-encoded frames.
    
    Use this for offline playback or when you need all frames at once.
    Warning: Response can be large for long sentences.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        service = get_text_to_sign_service()
        result = service.get_animation_data(text)
        
        animations = []
        for anim in result["animations"]:
            frames = [FrameData(**f) for f in anim["frames"]]
            animations.append(AnimationWord(
                word=anim["word"],
                found=anim["found"],
                frames=frames
            ))
        
        return AnimationResponse(
            success=True,
            original_text=result["original_text"],
            animations=animations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frames/{word}/{filename}")
async def get_frame_image(word: str, filename: str):
    """
    Get a specific frame image file.
    
    Path format: /frames/{WORD_FOLDER}/{filename}
    Example: /frames/HELLO_HI/HELLO HI-1.jpg
    """
    service = get_text_to_sign_service()
    
    # Find the word folder
    folder = service.find_word(word)
    if folder is None:
        # Try exact folder name
        exact_folder = FRAMES_WORD_LEVEL / word
        if exact_folder.exists():
            folder = exact_folder
        else:
            raise HTTPException(status_code=404, detail=f"Word not found: {word}")
    
    # Find the file
    file_path = folder / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame not found: {filename}")
    
    return FileResponse(
        file_path,
        media_type="image/jpeg",
        filename=filename
    )


@router.get("/word/{word}")
async def get_word_frames(word: str):
    """
    Get all frame paths for a specific word.
    
    Returns list of paths that can be used with /frames endpoint.
    """
    service = get_text_to_sign_service()
    
    paths = service.get_frame_paths_for_word(word)
    if not paths:
        raise HTTPException(status_code=404, detail=f"No frames found for: {word}")
    
    return {
        "word": word,
        "frames_count": len(paths),
        "frame_paths": paths
    }
