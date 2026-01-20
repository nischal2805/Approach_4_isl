"""
ISL Backend - Text-to-Sign Service

Converts English text to ISL sign animations using pre-recorded frame sequences.
"""
import os
import re
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from config import FRAMES_WORD_LEVEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextToSignService:
    """Service for converting text to ISL sign animations."""
    
    def __init__(self):
        self.word_to_folder: Dict[str, Path] = {}
        self.available_words: List[str] = []
        self._loaded = False
        
    def load(self) -> bool:
        """Load available sign word mappings."""
        try:
            logger.info(f"Loading Text-to-Sign mappings from {FRAMES_WORD_LEVEL}")
            
            if not FRAMES_WORD_LEVEL.exists():
                logger.error(f"Frames directory not found: {FRAMES_WORD_LEVEL}")
                return False
            
            # Map each folder to normalized word(s)
            for folder in FRAMES_WORD_LEVEL.iterdir():
                if folder.is_dir():
                    folder_name = folder.name
                    # Normalize: "HELLO_HI" -> ["hello", "hi"]
                    # "DON'T CARE" -> ["don't", "care", "dont"]
                    words = self._normalize_folder_name(folder_name)
                    for word in words:
                        self.word_to_folder[word] = folder
                    self.available_words.append(folder_name)
            
            self._loaded = True
            logger.info(f"Loaded {len(self.available_words)} sign words")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
            return False
    
    def _normalize_folder_name(self, name: str) -> List[str]:
        """Convert folder name to searchable words."""
        words = []
        
        # Original lowercase
        words.append(name.lower())
        
        # Split by underscore, space, or /
        parts = re.split(r'[_/\s]+', name.lower())
        words.extend(parts)
        
        # Remove apostrophes for matching
        for part in parts:
            cleaned = part.replace("'", "")
            if cleaned != part:
                words.append(cleaned)
        
        return list(set(words))
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_available_words(self) -> List[str]:
        """Return list of available sign words."""
        return sorted(self.available_words)
    
    def find_word(self, word: str) -> Optional[Path]:
        """Find the folder for a given word."""
        if not self._loaded:
            self.load()
        
        normalized = word.lower().strip()
        
        # Direct match
        if normalized in self.word_to_folder:
            return self.word_to_folder[normalized]
        
        # Try without punctuation
        cleaned = re.sub(r'[^\w\s]', '', normalized)
        if cleaned in self.word_to_folder:
            return self.word_to_folder[cleaned]
        
        return None
    
    def get_frames_for_word(self, word: str) -> List[Dict[str, Any]]:
        """
        Get all frame images for a word.
        Returns list of {filename, data_base64} sorted by frame number.
        """
        folder = self.find_word(word)
        if folder is None:
            return []
        
        frames = []
        for img_file in sorted(folder.glob("*.jpg")):
            with open(img_file, "rb") as f:
                img_data = f.read()
            frames.append({
                "filename": img_file.name,
                "data_base64": base64.b64encode(img_data).decode("utf-8")
            })
        
        # Also check for png
        for img_file in sorted(folder.glob("*.png")):
            with open(img_file, "rb") as f:
                img_data = f.read()
            frames.append({
                "filename": img_file.name,
                "data_base64": base64.b64encode(img_data).decode("utf-8")
            })
        
        return frames
    
    def get_frame_paths_for_word(self, word: str) -> List[str]:
        """Get file paths for frames (for serving via static files)."""
        folder = self.find_word(word)
        if folder is None:
            return []
        
        paths = []
        for img_file in sorted(folder.glob("*.jpg")):
            # Return relative path from FRAMES_WORD_LEVEL
            rel_path = img_file.relative_to(FRAMES_WORD_LEVEL)
            paths.append(str(rel_path).replace("\\", "/"))
        
        for img_file in sorted(folder.glob("*.png")):
            rel_path = img_file.relative_to(FRAMES_WORD_LEVEL)
            paths.append(str(rel_path).replace("\\", "/"))
        
        return paths
    
    def text_to_signs(self, text: str) -> Dict[str, Any]:
        """
        Convert a sentence to ISL signs.
        
        Args:
            text: English sentence
            
        Returns:
            {
                "original_text": str,
                "words": List[{word, found, frames_count, frame_paths}],
                "found_count": int,
                "total_count": int
            }
        """
        if not self._loaded:
            self.load()
        
        # Simple tokenization - split by spaces and punctuation
        words_raw = re.findall(r'\b\w+\b', text.lower())
        
        results = []
        found_count = 0
        
        for word in words_raw:
            frame_paths = self.get_frame_paths_for_word(word)
            found = len(frame_paths) > 0
            if found:
                found_count += 1
            
            results.append({
                "word": word,
                "found": found,
                "frames_count": len(frame_paths),
                "frame_paths": frame_paths
            })
        
        return {
            "original_text": text,
            "words": results,
            "found_count": found_count,
            "total_count": len(words_raw)
        }
    
    def get_animation_data(self, text: str) -> Dict[str, Any]:
        """
        Get complete animation data for a sentence.
        Includes base64-encoded frames for each word.
        """
        if not self._loaded:
            self.load()
        
        words_raw = re.findall(r'\b\w+\b', text.lower())
        
        animations = []
        
        for word in words_raw:
            frames = self.get_frames_for_word(word)
            animations.append({
                "word": word,
                "found": len(frames) > 0,
                "frames": frames
            })
        
        return {
            "original_text": text,
            "animations": animations
        }


# Singleton instance
_service: Optional[TextToSignService] = None

def get_text_to_sign_service() -> TextToSignService:
    global _service
    if _service is None:
        _service = TextToSignService()
        _service.load()
    return _service
