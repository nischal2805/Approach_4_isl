#!/usr/bin/env python3
"""
Generate NO_SIGN class data for pause detection.

This script generates synthetic "no sign" samples to help the model
detect when the user is not signing (hands down, out of frame, or idle).

Usage:
    python generate_no_sign.py --output data/INCLUDE50/NO_SIGN --count 200
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import (
    LANDMARKS_PER_FRAME,
    TOTAL_FEATURES,
    aggregate_temporal_features
)


def generate_no_sign_samples(
    num_samples: int = 200,
    num_frames_range: tuple = (30, 90),
    output_dir: str = None
) -> np.ndarray:
    """
    Generate NO_SIGN class samples.
    
    These simulate scenarios where hands are not visible or user is idle:
    1. All zeros (hands not detected)
    2. Very low movement (static pose)
    3. Random noise (garbage frames)
    
    Args:
        num_samples: Number of samples to generate
        num_frames_range: Range of frames per sample
        output_dir: Optional directory to save samples
    
    Returns:
        Array of shape [num_samples, 675] with aggregated features
    """
    print(f"Generating {num_samples} NO_SIGN samples...")
    
    np.random.seed(42)
    samples = []
    
    for i in range(num_samples):
        num_frames = np.random.randint(num_frames_range[0], num_frames_range[1])
        
        # Choose generation strategy
        strategy = np.random.choice(['zeros', 'static', 'noise', 'partial'], 
                                    p=[0.3, 0.3, 0.2, 0.2])
        
        if strategy == 'zeros':
            # Completely empty - hands not detected at all
            landmarks = np.zeros((num_frames, LANDMARKS_PER_FRAME), dtype=np.float32)
            # Add tiny noise to avoid division by zero issues
            landmarks += np.random.randn(*landmarks.shape).astype(np.float32) * 0.001
            
        elif strategy == 'static':
            # Static pose - very low variance (person standing still)
            base_pose = np.random.rand(LANDMARKS_PER_FRAME).astype(np.float32) * 0.5 + 0.25
            landmarks = np.tile(base_pose, (num_frames, 1))
            # Very small movement
            landmarks += np.random.randn(*landmarks.shape).astype(np.float32) * 0.01
            
        elif strategy == 'noise':
            # Random noise - garbage detection
            landmarks = np.random.rand(num_frames, LANDMARKS_PER_FRAME).astype(np.float32)
            
        else:  # partial
            # Partial detection - only pose, no hands
            landmarks = np.zeros((num_frames, LANDMARKS_PER_FRAME), dtype=np.float32)
            # Only fill pose landmarks (first 99 values = 33 landmarks * 3)
            pose_features = 33 * 3  # 99
            base_pose = np.random.rand(pose_features).astype(np.float32) * 0.5 + 0.25
            for j in range(num_frames):
                landmarks[j, :pose_features] = base_pose + np.random.randn(pose_features).astype(np.float32) * 0.02
        
        # Aggregate features
        try:
            aggregated = aggregate_temporal_features(landmarks)
            samples.append(aggregated)
        except Exception as e:
            print(f"Warning: Failed to aggregate sample {i}: {e}")
            continue
    
    samples = np.array(samples, dtype=np.float32)
    print(f"Generated {len(samples)} NO_SIGN samples with shape {samples.shape}")
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as individual .npy files to match video extraction format
        # Or save as single file for direct use
        np.save(output_path / 'no_sign_features.npy', samples)
        print(f"Saved to: {output_path / 'no_sign_features.npy'}")
    
    return samples


def add_no_sign_to_processed_data(
    processed_dir: str,
    num_samples: int = 200
):
    """
    Add NO_SIGN class to already processed dataset.
    
    Args:
        processed_dir: Path to processed data directory
        num_samples: Number of NO_SIGN samples to add
    """
    import joblib
    from sklearn.preprocessing import LabelEncoder
    
    processed_path = Path(processed_dir)
    
    # Load existing data
    X_train = np.load(processed_path / 'X_train.npy')
    X_val = np.load(processed_path / 'X_val.npy')
    X_test = np.load(processed_path / 'X_test.npy')
    y_train = np.load(processed_path / 'y_train.npy')
    y_val = np.load(processed_path / 'y_val.npy')
    y_test = np.load(processed_path / 'y_test.npy')
    label_encoder = joblib.load(processed_path / 'label_encoder.pkl')
    scaler = joblib.load(processed_path / 'scaler.pkl')
    
    print(f"Original dataset: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"Original classes: {len(label_encoder.classes_)}")
    
    # Generate NO_SIGN samples
    no_sign_samples = generate_no_sign_samples(num_samples)
    
    # Scale using existing scaler
    no_sign_scaled = scaler.transform(no_sign_samples)
    
    # Split NO_SIGN samples into train/val/test (70/15/15)
    n_train = int(0.7 * len(no_sign_scaled))
    n_val = int(0.15 * len(no_sign_scaled))
    
    no_sign_train = no_sign_scaled[:n_train]
    no_sign_val = no_sign_scaled[n_train:n_train + n_val]
    no_sign_test = no_sign_scaled[n_train + n_val:]
    
    # Create new label encoder with NO_SIGN
    new_classes = list(label_encoder.classes_) + ['NO_SIGN']
    new_label_encoder = LabelEncoder()
    new_label_encoder.fit(new_classes)
    
    no_sign_label = new_label_encoder.transform(['NO_SIGN'])[0]
    
    # Add to datasets
    X_train_new = np.vstack([X_train, no_sign_train])
    X_val_new = np.vstack([X_val, no_sign_val])
    X_test_new = np.vstack([X_test, no_sign_test])
    
    y_train_new = np.concatenate([y_train, np.full(len(no_sign_train), no_sign_label)])
    y_val_new = np.concatenate([y_val, np.full(len(no_sign_val), no_sign_label)])
    y_test_new = np.concatenate([y_test, np.full(len(no_sign_test), no_sign_label)])
    
    # Shuffle
    train_idx = np.random.permutation(len(X_train_new))
    val_idx = np.random.permutation(len(X_val_new))
    test_idx = np.random.permutation(len(X_test_new))
    
    X_train_new = X_train_new[train_idx]
    y_train_new = y_train_new[train_idx]
    X_val_new = X_val_new[val_idx]
    y_val_new = y_val_new[val_idx]
    X_test_new = X_test_new[test_idx]
    y_test_new = y_test_new[test_idx]
    
    # Save updated data
    np.save(processed_path / 'X_train.npy', X_train_new)
    np.save(processed_path / 'X_val.npy', X_val_new)
    np.save(processed_path / 'X_test.npy', X_test_new)
    np.save(processed_path / 'y_train.npy', y_train_new)
    np.save(processed_path / 'y_val.npy', y_val_new)
    np.save(processed_path / 'y_test.npy', y_test_new)
    
    joblib.dump(new_label_encoder, processed_path / 'label_encoder.pkl')
    
    # Update labels.txt
    with open(processed_path / 'labels.txt', 'w') as f:
        for cls in new_label_encoder.classes_:
            f.write(f"{cls}\n")
    
    print(f"\nUpdated dataset:")
    print(f"  Train: {len(X_train_new)} (added {len(no_sign_train)} NO_SIGN)")
    print(f"  Val: {len(X_val_new)} (added {len(no_sign_val)} NO_SIGN)")
    print(f"  Test: {len(X_test_new)} (added {len(no_sign_test)} NO_SIGN)")
    print(f"  Classes: {len(new_label_encoder.classes_)} (+1 NO_SIGN)")


def main():
    parser = argparse.ArgumentParser(description='Generate NO_SIGN class data')
    parser.add_argument('--output', type=str, default='data/no_sign',
                       help='Output directory for NO_SIGN samples')
    parser.add_argument('--count', type=int, default=200,
                       help='Number of NO_SIGN samples to generate')
    parser.add_argument('--add-to-processed', type=str, default=None,
                       help='Path to processed data directory to add NO_SIGN class')
    
    args = parser.parse_args()
    
    if args.add_to_processed:
        add_no_sign_to_processed_data(args.add_to_processed, args.count)
    else:
        generate_no_sign_samples(args.count, output_dir=args.output)


if __name__ == '__main__':
    main()
