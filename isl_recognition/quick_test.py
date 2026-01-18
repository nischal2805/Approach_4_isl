#!/usr/bin/env python3
"""
Quick Pipeline Test

Tests the entire training pipeline using synthetic data or existing data.
Run this to verify everything works before processing the full INCLUDE-50 dataset.

Usage:
    python quick_test.py                    # Test with synthetic data
    python quick_test.py --use-existing     # Test with existing processed data
"""

import os
import sys
import argparse
import time
import tempfile
from pathlib import Path
import numpy as np


def test_with_synthetic_data():
    """Test pipeline with synthetic data."""
    print("=" * 60)
    print("Testing Pipeline with Synthetic Data")
    print("=" * 60)
    
    from data_preprocessing import create_synthetic_dataset, aggregate_temporal_features, TOTAL_FEATURES
    from train_model import train_with_synthetic_data
    
    # Test 1: Feature aggregation
    print("\n[Test 1] Feature Aggregation...")
    test_landmarks = np.random.rand(30, 225).astype(np.float32)
    aggregated = aggregate_temporal_features(test_landmarks)
    assert aggregated.shape == (TOTAL_FEATURES,), f"Expected {TOTAL_FEATURES}, got {aggregated.shape}"
    print(f"  ✓ Aggregated {test_landmarks.shape} → {aggregated.shape}")
    
    # Test 2: Synthetic dataset
    print("\n[Test 2] Synthetic Dataset Generation...")
    X, y, classes = create_synthetic_dataset(num_classes=5, samples_per_class=10)
    assert X.shape == (50, TOTAL_FEATURES), f"Unexpected shape: {X.shape}"
    print(f"  ✓ Generated dataset: {X.shape}, classes: {len(classes)}")
    
    # Test 3: Training
    print("\n[Test 3] Training with Synthetic Data...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_with_synthetic_data(
            num_classes=5,
            samples_per_class=20,
            output_folder=tmp_dir
        )
        
        # Check outputs
        assert os.path.exists(os.path.join(tmp_dir, 'model.pkl')), "model.pkl not found"
        assert os.path.exists(os.path.join(tmp_dir, 'label_encoder.pkl')), "label_encoder.pkl not found"
        print(f"  ✓ Model trained and saved")
    
    # Test 4: NO_SIGN generation
    print("\n[Test 4] NO_SIGN Sample Generation...")
    from generate_no_sign import generate_no_sign_samples
    no_sign = generate_no_sign_samples(num_samples=20)
    assert no_sign.shape == (20, TOTAL_FEATURES), f"Unexpected shape: {no_sign.shape}"
    print(f"  ✓ Generated NO_SIGN samples: {no_sign.shape}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


def test_with_existing_data(processed_dir: str):
    """Test pipeline with existing processed data."""
    print("=" * 60)
    print("Testing Pipeline with Existing Data")
    print("=" * 60)
    
    from data_preprocessing import load_processed_data
    import joblib
    
    # Load data
    print("\n[Test 1] Loading Processed Data...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler = load_processed_data(processed_dir)
    print(f"  ✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  ✓ Classes: {len(label_encoder.classes_)}")
    
    # Check for NaN/Inf
    print("\n[Test 2] Checking Data Quality...")
    assert not np.isnan(X_train).any(), "NaN values in training data"
    assert not np.isinf(X_train).any(), "Inf values in training data"
    print(f"  ✓ No NaN/Inf values")
    
    # Class distribution
    print("\n[Test 3] Class Distribution...")
    unique, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()
    max_count = counts.max()
    print(f"  ✓ Classes: {len(unique)}, Min/Max samples: {min_count}/{max_count}")
    
    print("\n" + "=" * 60)
    print("DATA VALIDATION PASSED ✓")
    print("=" * 60)


def test_real_time_inference():
    """Test real-time inference module."""
    print("\n" + "=" * 60)
    print("Testing Real-Time Inference (no camera)")
    print("=" * 60)
    
    models_dir = Path(__file__).parent / 'models'
    
    if not (models_dir / 'model.pkl').exists():
        print("  ⚠ No model found, skipping inference test")
        return
    
    import joblib
    
    model = joblib.load(models_dir / 'model.pkl')
    scaler = joblib.load(models_dir / 'scaler.pkl')
    label_encoder = joblib.load(models_dir / 'label_encoder.pkl')
    
    # Test prediction
    dummy_features = np.random.rand(1, 675).astype(np.float32)
    dummy_scaled = scaler.transform(dummy_features)
    prediction = model.predict(dummy_scaled)
    proba = model.predict_proba(dummy_scaled)
    
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    confidence = proba.max()
    
    print(f"  ✓ Test prediction: '{predicted_class}' ({confidence:.2%} confidence)")
    
    print("\n" + "=" * 60)
    print("INFERENCE TEST PASSED ✓")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Quick Pipeline Test')
    parser.add_argument('--use-existing', type=str, default=None,
                       help='Path to existing processed data')
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip inference test')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.use_existing:
        test_with_existing_data(args.use_existing)
    else:
        test_with_synthetic_data()
    
    if not args.skip_inference:
        test_real_time_inference()
    
    elapsed = time.time() - start_time
    print(f"\nTotal test time: {elapsed:.1f} seconds")


if __name__ == '__main__':
    main()
