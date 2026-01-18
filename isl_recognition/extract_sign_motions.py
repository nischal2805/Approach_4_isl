#!/usr/bin/env python3
"""
Extract Sign Motions from INCLUDE-50 Videos

This script extracts MediaPipe landmarks from sign language videos
and saves them as JSON files for playback in the Flutter app.

Usage:
    python extract_sign_motions.py --input data/INCLUDE50 --output ../isl_app/assets/signs/words
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe setup
mp_holistic = mp.solutions.holistic


def extract_landmarks_from_video(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract pose and hand landmarks from a video file.
    
    Returns:
        Dictionary with frames containing pose and hand landmark data
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default fallback
    
    frames_data = []
    
    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                
                # Extract landmarks
                frame_data = {
                    "timestamp": frame_idx / fps,
                    "pose": None,
                    "left_hand": None,
                    "right_hand": None
                }
                
                # Pose landmarks (33 points)
                if results.pose_landmarks:
                    frame_data["pose"] = [
                        [lm.x, lm.y, lm.z] 
                        for lm in results.pose_landmarks.landmark
                    ]
                
                # Left hand landmarks (21 points)
                if results.left_hand_landmarks:
                    frame_data["left_hand"] = [
                        [lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark
                    ]
                
                # Right hand landmarks (21 points)
                if results.right_hand_landmarks:
                    frame_data["right_hand"] = [
                        [lm.x, lm.y, lm.z]
                        for lm in results.right_hand_landmarks.landmark
                    ]
                
                # Only add frames with at least pose detected
                if frame_data["pose"] is not None:
                    frames_data.append(frame_data)
                
                frame_idx += 1
    
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return None
    
    finally:
        cap.release()
    
    if len(frames_data) == 0:
        logger.warning(f"No landmarks detected in: {video_path}")
        return None
    
    return {
        "fps": fps,
        "frame_count": len(frames_data),
        "frames": frames_data
    }


def process_dataset(input_dir: str, output_dir: str, sign_type: str = "word"):
    """
    Process all videos in a dataset folder and save as JSON files.
    
    Args:
        input_dir: Path to dataset (e.g., INCLUDE50 with class folders)
        output_dir: Path to save JSON files
        sign_type: "word", "letter", or "number"
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI'}
    
    # Find all class folders
    class_folders = [f for f in input_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    logger.info(f"Found {len(class_folders)} sign classes")
    
    processed = 0
    failed = 0
    
    for class_folder in tqdm(class_folders, desc="Processing signs"):
        class_name = class_folder.name
        
        # Find videos in this class
        videos = [f for f in class_folder.iterdir() if f.suffix.lower() in video_extensions]
        
        if not videos:
            logger.warning(f"No videos found in: {class_folder}")
            continue
        
        # Use first video (or best quality one)
        video_path = videos[0]
        
        # Extract landmarks
        result = extract_landmarks_from_video(str(video_path))
        
        if result is None:
            failed += 1
            continue
        
        # Build output JSON
        output_data = {
            "label": class_name,
            "type": sign_type,
            "fps": result["fps"],
            "frame_count": result["frame_count"],
            "frames": result["frames"]
        }
        
        # Sanitize filename
        safe_name = class_name.lower().replace(" ", "_").replace(".", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        
        output_file = output_path / f"{safe_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f)
        
        processed += 1
    
    logger.info(f"\nProcessed: {processed}, Failed: {failed}")
    
    # Create index file
    create_index(output_path, sign_type)


def create_index(output_dir: Path, sign_type: str):
    """Create an index JSON file listing all available signs."""
    signs = []
    
    for json_file in output_dir.glob("*.json"):
        if json_file.name == "index.json":
            continue
        
        with open(json_file) as f:
            data = json.load(f)
            signs.append({
                "label": data["label"],
                "file": json_file.name,
                "frame_count": data.get("frame_count", 0)
            })
    
    index = {
        "type": sign_type,
        "count": len(signs),
        "signs": sorted(signs, key=lambda x: x["label"])
    }
    
    with open(output_dir / "index.json", 'w') as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"Created index.json with {len(signs)} signs")


def main():
    parser = argparse.ArgumentParser(description='Extract sign motions from videos')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to dataset folder (e.g., INCLUDE50)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for JSON files')
    parser.add_argument('--type', type=str, default='word',
                       choices=['word', 'letter', 'number'],
                       help='Type of signs being processed')
    
    args = parser.parse_args()
    
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Type: {args.type}")
    
    process_dataset(args.input, args.output, args.type)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
