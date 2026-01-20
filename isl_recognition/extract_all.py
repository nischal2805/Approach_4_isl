#!/usr/bin/env python3
"""
Unified Landmark Extraction with Multicore Support

Extracts landmarks from videos and saves:
1. Aggregated features (.npy) for Sign→Text training
2. Raw frame data (.json) for Text→Sign playback

Usage:
    python extract_all.py --input data/INCLUDE50 --output-training data/processed --output-signs ../isl_app/assets/signs/words --workers 8
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import cv2
import numpy as np
import mediapipe as holistic_mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
LANDMARKS_PER_FRAME = (NUM_POSE_LANDMARKS + 2 * NUM_HAND_LANDMARKS) * 3  # 225


def extract_from_video(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract landmarks from a single video.
    Returns both raw frames (for playback) and aggregated features (for training).
    """
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    all_frames = []  # For playback JSON
    all_landmarks = []  # For training
    
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
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                
                # Extract pose landmarks
                pose_data = None
                left_hand_data = None
                right_hand_data = None
                frame_landmarks = np.zeros(LANDMARKS_PER_FRAME, dtype=np.float32)
                
                if results.pose_landmarks:
                    pose_data = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        idx = i * 3
                        frame_landmarks[idx] = lm.x
                        frame_landmarks[idx + 1] = lm.y
                        frame_landmarks[idx + 2] = lm.z
                
                pose_offset = NUM_POSE_LANDMARKS * 3
                
                if results.left_hand_landmarks:
                    left_hand_data = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                    for i, lm in enumerate(results.left_hand_landmarks.landmark):
                        idx = pose_offset + i * 3
                        frame_landmarks[idx] = lm.x
                        frame_landmarks[idx + 1] = lm.y
                        frame_landmarks[idx + 2] = lm.z
                
                left_hand_offset = pose_offset + NUM_HAND_LANDMARKS * 3
                
                if results.right_hand_landmarks:
                    right_hand_data = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                    for i, lm in enumerate(results.right_hand_landmarks.landmark):
                        idx = left_hand_offset + i * 3
                        frame_landmarks[idx] = lm.x
                        frame_landmarks[idx + 1] = lm.y
                        frame_landmarks[idx + 2] = lm.z
                
                # Save for playback (JSON format)
                if pose_data is not None:
                    all_frames.append({
                        "timestamp": frame_idx / fps,
                        "pose": pose_data,
                        "left_hand": left_hand_data,
                        "right_hand": right_hand_data
                    })
                
                # Save for training
                all_landmarks.append(frame_landmarks)
                frame_idx += 1
    
    except Exception as e:
        return None
    
    finally:
        cap.release()
    
    if len(all_landmarks) == 0:
        return None
    
    # Aggregate for training
    landmarks_array = np.array(all_landmarks)
    mean_features = np.mean(landmarks_array, axis=0)
    max_features = np.max(landmarks_array, axis=0)
    std_features = np.std(landmarks_array, axis=0)
    aggregated = np.concatenate([mean_features, max_features, std_features])
    aggregated = np.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)
    
    return {
        "fps": fps,
        "frames": all_frames,  # For Text→Sign playback
        "aggregated": aggregated.astype(np.float32),  # For Sign→Text training
        "frame_count": len(all_frames)
    }


def process_video_wrapper(args: Tuple[str, str]) -> Tuple[str, str, Optional[Dict]]:
    """Wrapper for multiprocessing."""
    video_path, label = args
    result = extract_from_video(video_path)
    return video_path, label, result


def process_dataset_multicore(
    dataset_folder: str,
    output_training: str,
    output_signs: str = None,
    num_workers: int = None,
    test_size: float = 0.15,
    val_size: float = 0.15
):
    """
    Process dataset using multiple CPU cores.
    Saves both training data and playback JSON in one pass.
    """
    dataset_path = Path(dataset_folder)
    training_path = Path(output_training)
    training_path.mkdir(parents=True, exist_ok=True)
    
    if output_signs:
        signs_path = Path(output_signs)
        signs_path.mkdir(parents=True, exist_ok=True)
    
    # Determine worker count
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)
    
    logger.info(f"Using {num_workers} worker processes")
    
    # Find all videos - handles nested structure: Category/SignName/videos
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV'}
    videos_with_labels = []
    
    # Check if this is nested structure (Category/SignName) or flat (SignName)
    for item in dataset_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this folder contains videos directly
            has_videos = any(f.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'} 
                           for f in item.iterdir() if f.is_file())
            
            if has_videos:
                # Flat structure: dataset/SignName/videos
                class_name = item.name
                # Extract clean label (remove number prefix like "48. Hello" -> "Hello")
                if '. ' in class_name:
                    class_name = class_name.split('. ', 1)[1]
                
                for video_file in item.iterdir():
                    if video_file.is_file() and video_file.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
                        videos_with_labels.append((str(video_file), class_name))
            else:
                # Nested structure: dataset/Category/SignName/videos
                category = item.name
                for sign_folder in item.iterdir():
                    if sign_folder.is_dir() and not sign_folder.name.startswith('.'):
                        class_name = sign_folder.name
                        # Extract clean label (remove number prefix like "48. Hello" -> "Hello")
                        if '. ' in class_name:
                            class_name = class_name.split('. ', 1)[1]
                        
                        for video_file in sign_folder.iterdir():
                            if video_file.is_file() and video_file.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
                                videos_with_labels.append((str(video_file), class_name))
    
    logger.info(f"Found {len(videos_with_labels)} videos")
    
    # Process with multiprocessing
    features = []
    labels = []
    sign_data = {}
    failed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_video_wrapper, args): args for args in videos_with_labels}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            try:
                video_path, label, result = future.result()
                
                if result is None:
                    failed += 1
                    continue
                
                # Save for training
                features.append(result["aggregated"])
                labels.append(label)
                
                # Save for playback (one JSON per class)
                if output_signs and label not in sign_data:
                    sign_data[label] = {
                        "label": label,
                        "type": "word",
                        "fps": result["fps"],
                        "frame_count": result["frame_count"],
                        "frames": result["frames"]
                    }
            
            except Exception as e:
                failed += 1
    
    logger.info(f"Processed: {len(features)}, Failed: {failed}")
    
    # Save training data
    if len(features) > 0:
        X = np.array(features, dtype=np.float32)
        y = np.array(labels)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Save
        np.save(training_path / 'X_train.npy', X_train)
        np.save(training_path / 'X_val.npy', X_val)
        np.save(training_path / 'X_test.npy', X_test)
        np.save(training_path / 'y_train.npy', y_train)
        np.save(training_path / 'y_val.npy', y_val)
        np.save(training_path / 'y_test.npy', y_test)
        joblib.dump(label_encoder, training_path / 'label_encoder.pkl')
        joblib.dump(scaler, training_path / 'scaler.pkl')
        
        # Save labels.txt
        with open(training_path / 'labels.txt', 'w') as f:
            for label in label_encoder.classes_:
                f.write(f"{label}\n")
        
        logger.info(f"Training data saved to: {training_path}")
    
    # Save sign JSONs for playback
    if output_signs and sign_data:
        for label, data in sign_data.items():
            safe_name = label.lower().replace(" ", "_").replace(".", "_")
            safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
            
            with open(signs_path / f"{safe_name}.json", 'w') as f:
                json.dump(data, f)
        
        # Create index
        index = {
            "type": "word",
            "count": len(sign_data),
            "signs": [{"label": d["label"], "file": f"{d['label'].lower().replace(' ', '_')}.json"} 
                     for d in sign_data.values()]
        }
        with open(signs_path / "index.json", 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Sign JSONs saved to: {signs_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified multicore landmark extraction')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to dataset folder')
    parser.add_argument('--output-training', type=str, default='data/processed',
                       help='Output for training data')
    parser.add_argument('--output-signs', type=str, default=None,
                       help='Output for sign playback JSONs (optional)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count - 2)')
    
    args = parser.parse_args()
    
    process_dataset_multicore(
        args.input,
        args.output_training,
        args.output_signs,
        args.workers
    )
    
    logger.info("Done!")


if __name__ == '__main__':
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
