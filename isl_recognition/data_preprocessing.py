#!/usr/bin/env python3
"""
ISL Recognition System - Data Preprocessing Module

This module handles:
1. Extracting MediaPipe landmarks from video files
2. Temporal feature aggregation (mean, max, std)
3. Dataset processing with train/val/test splits

Author: ISL Recognition Team
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MediaPipe configuration
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Constants
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
LANDMARKS_PER_FRAME = (NUM_POSE_LANDMARKS + 2 * NUM_HAND_LANDMARKS) * 3  # 225
TEMPORAL_FEATURES = 3  # mean, max, std
TOTAL_FEATURES = LANDMARKS_PER_FRAME * TEMPORAL_FEATURES  # 675


def extract_landmarks_from_video(
    video_path: str,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
) -> Optional[np.ndarray]:
    """
    Extract MediaPipe Holistic landmarks from a video file.
    
    Args:
        video_path: Path to the video file
        min_detection_confidence: Minimum detection confidence threshold
        min_tracking_confidence: Minimum tracking confidence threshold
    
    Returns:
        numpy array of shape [num_frames, 225] or None if extraction fails
        Each frame has: 33 pose + 21 left hand + 21 right hand = 75 landmarks
        Each landmark has (x, y, z) = 225 features per frame
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None
    
    all_landmarks = []
    frames_with_detection = 0
    total_frames = 0
    
    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        ) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                
                # Extract landmarks for this frame
                frame_landmarks = extract_frame_landmarks(results)
                
                if frame_landmarks is not None:
                    all_landmarks.append(frame_landmarks)
                    frames_with_detection += 1
    
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return None
    
    finally:
        cap.release()
    
    # Check if we got enough valid frames
    if len(all_landmarks) == 0:
        logger.warning(f"No landmarks detected in video: {video_path}")
        return None
    
    detection_rate = frames_with_detection / max(total_frames, 1)
    if detection_rate < 0.3:  # Less than 30% frames have detections
        logger.warning(f"Low detection rate ({detection_rate:.2%}) for: {video_path}")
    
    landmarks_array = np.array(all_landmarks, dtype=np.float32)
    logger.debug(f"Extracted {len(all_landmarks)} frames from {video_path}")
    
    return landmarks_array


def extract_frame_landmarks(results) -> Optional[np.ndarray]:
    """
    Extract landmarks from a single frame's MediaPipe results.
    
    Args:
        results: MediaPipe Holistic results object
    
    Returns:
        numpy array of shape [225] or None if no detection
    """
    landmarks = []
    
    # Extract pose landmarks (33 landmarks × 3 coords = 99 features)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        # Fill with zeros if no pose detected
        landmarks.extend([0.0] * (NUM_POSE_LANDMARKS * 3))
    
    # Extract left hand landmarks (21 landmarks × 3 coords = 63 features)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (NUM_HAND_LANDMARKS * 3))
    
    # Extract right hand landmarks (21 landmarks × 3 coords = 63 features)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (NUM_HAND_LANDMARKS * 3))
    
    # Validate shape
    if len(landmarks) != LANDMARKS_PER_FRAME:
        logger.error(f"Unexpected landmark count: {len(landmarks)}, expected {LANDMARKS_PER_FRAME}")
        return None
    
    # Check if at least hands or pose are detected (not all zeros)
    landmarks_array = np.array(landmarks, dtype=np.float32)
    if np.all(landmarks_array == 0):
        return None
    
    return landmarks_array


def aggregate_temporal_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Aggregate temporal features across all frames using mean, max, and std.
    
    Args:
        landmarks: numpy array of shape [num_frames, 225]
    
    Returns:
        numpy array of shape [675] (225 × 3 aggregation methods)
    """
    if landmarks is None or len(landmarks) == 0:
        raise ValueError("Empty landmarks array provided")
    
    # Ensure 2D array
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(1, -1)
    
    # Validate shape
    if landmarks.shape[1] != LANDMARKS_PER_FRAME:
        raise ValueError(f"Expected {LANDMARKS_PER_FRAME} features per frame, got {landmarks.shape[1]}")
    
    # Compute temporal statistics
    mean_features = np.mean(landmarks, axis=0)  # [225]
    max_features = np.max(landmarks, axis=0)    # [225]
    std_features = np.std(landmarks, axis=0)    # [225]
    
    # Concatenate all features
    aggregated = np.concatenate([mean_features, max_features, std_features])  # [675]
    
    # Handle NaN values
    aggregated = np.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)
    
    return aggregated.astype(np.float32)


def find_videos_in_dataset(dataset_folder: str) -> List[Tuple[str, str]]:
    """
    Find all video files in the dataset folder and extract their labels.
    
    Supports common dataset structures:
    1. INCLUDE-50 style: dataset_folder/class_name/video.mp4
    2. Flat with CSV: dataset_folder/videos/ + labels.csv
    
    Args:
        dataset_folder: Path to the dataset root folder
    
    Returns:
        List of (video_path, label) tuples
    """
    dataset_path = Path(dataset_folder)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    videos_with_labels = []
    
    # Check for CSV labels file
    csv_files = list(dataset_path.glob('*.csv'))
    
    if csv_files:
        # CSV-based labeling
        logger.info(f"Found CSV label file(s): {csv_files}")
        labels_df = pd.read_csv(csv_files[0])
        
        # Try common column names
        video_col = None
        label_col = None
        
        for col in labels_df.columns:
            col_lower = col.lower()
            if 'video' in col_lower or 'file' in col_lower or 'path' in col_lower:
                video_col = col
            if 'label' in col_lower or 'class' in col_lower or 'sign' in col_lower:
                label_col = col
        
        if video_col and label_col:
            for _, row in labels_df.iterrows():
                video_path = dataset_path / str(row[video_col])
                if video_path.exists():
                    videos_with_labels.append((str(video_path), str(row[label_col])))
    
    if not videos_with_labels:
        # Directory-based labeling (folder name = class)
        logger.info("Using directory-based labeling (folder name = class)")
        for class_folder in dataset_path.iterdir():
            if class_folder.is_dir() and not class_folder.name.startswith('.'):
                class_name = class_folder.name
                for video_file in class_folder.iterdir():
                    if video_file.suffix.lower() in video_extensions:
                        videos_with_labels.append((str(video_file), class_name))
    
    logger.info(f"Found {len(videos_with_labels)} videos in dataset")
    return videos_with_labels


def process_dataset(
    dataset_folder: str,
    output_folder: str = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process entire dataset: extract landmarks, aggregate features, create splits.
    
    Args:
        dataset_folder: Path to the dataset root folder
        output_folder: Path to save processed data (optional)
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info(f"Processing dataset from: {dataset_folder}")
    
    # Find all videos
    videos_with_labels = find_videos_in_dataset(dataset_folder)
    
    if len(videos_with_labels) == 0:
        raise ValueError(f"No videos found in {dataset_folder}")
    
    # Extract features from all videos
    features = []
    labels = []
    failed_videos = []
    
    logger.info("Extracting landmarks from videos...")
    for video_path, label in tqdm(videos_with_labels, desc="Processing videos"):
        try:
            # Extract landmarks
            landmarks = extract_landmarks_from_video(video_path)
            
            if landmarks is None or len(landmarks) == 0:
                failed_videos.append(video_path)
                continue
            
            # Aggregate temporal features
            aggregated = aggregate_temporal_features(landmarks)
            
            features.append(aggregated)
            labels.append(label)
            
        except Exception as e:
            logger.warning(f"Failed to process {video_path}: {str(e)}")
            failed_videos.append(video_path)
    
    logger.info(f"Successfully processed {len(features)}/{len(videos_with_labels)} videos")
    if failed_videos:
        logger.warning(f"Failed videos: {len(failed_videos)}")
    
    if len(features) == 0:
        raise ValueError("No features extracted from any video")
    
    # Convert to numpy arrays
    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")
    logger.info(f"Classes: {list(label_encoder.classes_)}")
    
    # Create train/val/test splits
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_trainval
    )
    
    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save processed data if output folder specified
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'X_train.npy', X_train)
        np.save(output_path / 'X_val.npy', X_val)
        np.save(output_path / 'X_test.npy', X_test)
        np.save(output_path / 'y_train.npy', y_train)
        np.save(output_path / 'y_val.npy', y_val)
        np.save(output_path / 'y_test.npy', y_test)
        
        joblib.dump(label_encoder, output_path / 'label_encoder.pkl')
        joblib.dump(scaler, output_path / 'scaler.pkl')
        
        # Save class names to text file
        with open(output_path / 'labels.txt', 'w') as f:
            for cls in label_encoder.classes_:
                f.write(f"{cls}\n")
        
        logger.info(f"Saved processed data to: {output_folder}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_processed_data(input_folder: str) -> Tuple[np.ndarray, ...]:
    """
    Load previously processed dataset.
    
    Args:
        input_folder: Path to folder containing processed data
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler
    """
    input_path = Path(input_folder)
    
    X_train = np.load(input_path / 'X_train.npy')
    X_val = np.load(input_path / 'X_val.npy')
    X_test = np.load(input_path / 'X_test.npy')
    y_train = np.load(input_path / 'y_train.npy')
    y_val = np.load(input_path / 'y_val.npy')
    y_test = np.load(input_path / 'y_test.npy')
    
    label_encoder = joblib.load(input_path / 'label_encoder.pkl')
    scaler = joblib.load(input_path / 'scaler.pkl')
    
    logger.info(f"Loaded processed data from: {input_folder}")
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler


def create_synthetic_dataset(
    num_classes: int = 10,
    samples_per_class: int = 20,
    num_frames_range: Tuple[int, int] = (30, 90)
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create synthetic dataset for testing purposes.
    
    Args:
        num_classes: Number of sign classes
        samples_per_class: Samples per class
        num_frames_range: Range of frames per video (min, max)
    
    Returns:
        X: Feature matrix [num_samples, 675]
        y: Labels [num_samples]
        class_names: List of class names
    """
    logger.info(f"Creating synthetic dataset: {num_classes} classes, {samples_per_class} samples each")
    
    X = []
    y = []
    class_names = [f"sign_{i:02d}" for i in range(num_classes)]
    
    np.random.seed(42)
    
    for class_idx in range(num_classes):
        # Create class-specific pattern
        class_pattern = np.random.randn(LANDMARKS_PER_FRAME) * 0.1 + class_idx * 0.05
        
        for _ in range(samples_per_class):
            # Random number of frames
            num_frames = np.random.randint(num_frames_range[0], num_frames_range[1])
            
            # Generate frame landmarks with class pattern + noise
            landmarks = np.random.randn(num_frames, LANDMARKS_PER_FRAME) * 0.05
            landmarks += class_pattern  # Add class-specific bias
            
            # Normalize to [0, 1] range like MediaPipe
            landmarks = (landmarks - landmarks.min()) / (landmarks.max() - landmarks.min() + 1e-8)
            
            # Aggregate features
            aggregated = aggregate_temporal_features(landmarks.astype(np.float32))
            
            X.append(aggregated)
            y.append(class_idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    logger.info(f"Synthetic dataset shape: {X.shape}")
    
    return X, y, class_names


def main():
    """Main entry point for data preprocessing."""
    parser = argparse.ArgumentParser(description='ISL Recognition Data Preprocessing')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to dataset folder or video file')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Path to save processed data')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Test set proportion (default: 0.15)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation set proportion (default: 0.15)')
    parser.add_argument('--single-video', action='store_true',
                        help='Process a single video file instead of dataset')
    
    args = parser.parse_args()
    
    if args.single_video:
        # Process single video
        logger.info(f"Processing single video: {args.input}")
        landmarks = extract_landmarks_from_video(args.input)
        
        if landmarks is not None:
            logger.info(f"Extracted landmarks shape: {landmarks.shape}")
            aggregated = aggregate_temporal_features(landmarks)
            logger.info(f"Aggregated features shape: {aggregated.shape}")
            
            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                np.save(output_path / 'landmarks.npy', landmarks)
                np.save(output_path / 'features.npy', aggregated)
                logger.info(f"Saved to: {args.output}")
        else:
            logger.error("Failed to extract landmarks from video")
            sys.exit(1)
    else:
        # Process full dataset
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = process_dataset(
                args.input,
                args.output,
                test_size=args.test_size,
                val_size=args.val_size
            )
            logger.info("Dataset processing completed successfully!")
            logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        except Exception as e:
            logger.error(f"Dataset processing failed: {str(e)}")
            sys.exit(1)


if __name__ == '__main__':
    main()
