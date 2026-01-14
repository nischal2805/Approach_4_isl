#!/usr/bin/env python3
"""
ISL Recognition System - Test Pipeline

Comprehensive tests for all components of the ISL recognition system.
"""

import os
import sys
import time
import logging
import argparse
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_landmark_extraction():
    """Test MediaPipe landmark extraction."""
    logger.info("TEST 1: Landmark Extraction")
    
    import cv2
    import mediapipe as mp
    
    # Create synthetic video frames
    mp_holistic = mp.solutions.holistic
    
    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        # Create a test frame with a visible person-like shape
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:400, 200:440] = [200, 180, 160]  # Skin-like color
        
        # Add face region
        cv2.circle(frame, (320, 150), 50, (200, 180, 160), -1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        # Even without real detection, we should be able to handle None results
        from data_preprocessing import extract_frame_landmarks
        landmarks = extract_frame_landmarks(results)
        
        # May be None or zeros if no detection, but should not error
        logger.info(f"  Landmark extraction function works (result: {'None' if landmarks is None else landmarks.shape})")
    
    logger.info("  ✓ Landmark extraction test passed")
    return True


def test_feature_aggregation():
    """Test temporal feature aggregation."""
    logger.info("TEST 2: Feature Aggregation")
    
    from data_preprocessing import aggregate_temporal_features, LANDMARKS_PER_FRAME, TOTAL_FEATURES
    
    # Create synthetic landmarks (30 frames, 225 features each)
    num_frames = 30
    landmarks = np.random.rand(num_frames, LANDMARKS_PER_FRAME).astype(np.float32)
    
    # Aggregate
    aggregated = aggregate_temporal_features(landmarks)
    
    assert aggregated.shape == (TOTAL_FEATURES,), f"Expected shape ({TOTAL_FEATURES},), got {aggregated.shape}"
    assert aggregated.dtype == np.float32, f"Expected float32, got {aggregated.dtype}"
    assert not np.any(np.isnan(aggregated)), "Contains NaN values"
    
    logger.info(f"  Input shape: {landmarks.shape}")
    logger.info(f"  Output shape: {aggregated.shape}")
    logger.info("  ✓ Feature aggregation test passed")
    return True


def test_synthetic_training():
    """Test model training with synthetic data."""
    logger.info("TEST 3: Synthetic Training")
    
    from data_preprocessing import create_synthetic_dataset
    from train_model import train_random_forest, evaluate_model
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Create synthetic data
    num_classes = 10
    samples_per_class = 20
    X, y, class_names = create_synthetic_dataset(num_classes, samples_per_class)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train (small model for speed)
    model = train_random_forest(X_train, y_train, n_estimators=20, max_depth=5)
    
    # Evaluate
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    metrics = evaluate_model(model, X_test, y_test, label_encoder, "Test")
    
    assert metrics['accuracy'] > 0.3, f"Accuracy too low: {metrics['accuracy']}"
    
    logger.info(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info("  ✓ Synthetic training test passed")
    return True


def test_model_save_load():
    """Test model save and load functionality."""
    logger.info("TEST 4: Model Save/Load")
    
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(50, 675)
        y = np.random.randint(0, 5, 50)
        model.fit(X, y)
        
        scaler = StandardScaler()
        scaler.fit(X)
        
        encoder = LabelEncoder()
        encoder.fit(['a', 'b', 'c', 'd', 'e'])
        
        joblib.dump(model, f"{tmpdir}/model.pkl")
        joblib.dump(scaler, f"{tmpdir}/scaler.pkl")
        joblib.dump(encoder, f"{tmpdir}/encoder.pkl")
        
        # Load and verify
        loaded_model = joblib.load(f"{tmpdir}/model.pkl")
        loaded_scaler = joblib.load(f"{tmpdir}/scaler.pkl")
        loaded_encoder = joblib.load(f"{tmpdir}/encoder.pkl")
        
        assert hasattr(loaded_model, 'predict')
        assert hasattr(loaded_scaler, 'transform')
        assert hasattr(loaded_encoder, 'inverse_transform')
    
    logger.info("  ✓ Model save/load test passed")
    return True


def test_tflite_conversion():
    """Test TFLite conversion via MLP distillation."""
    logger.info("TEST 5: TFLite Conversion")
    
    try:
        import tensorflow as tf
        from convert_to_mobile import create_mlp_model, convert_to_tflite
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create MLP
            input_dim = 675
            num_classes = 10
            mlp = create_mlp_model(input_dim, num_classes)
            
            # Quick training
            X = np.random.rand(100, input_dim).astype(np.float32)
            y = np.random.randint(0, num_classes, 100)
            mlp.fit(X, y, epochs=2, verbose=0)
            
            # Convert
            tflite_path = f"{tmpdir}/model.tflite"
            convert_to_tflite(mlp, tflite_path, quantize=True)
            
            assert os.path.exists(tflite_path)
            
            # Test inference
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_det = interpreter.get_input_details()
            output_det = interpreter.get_output_details()
            
            test_input = np.random.rand(1, input_dim).astype(np.float32)
            interpreter.set_tensor(input_det[0]['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_det[0]['index'])
            
            assert output.shape == (1, num_classes)
            
            logger.info(f"  TFLite model size: {os.path.getsize(tflite_path) / 1024:.2f} KB")
        
        logger.info("  ✓ TFLite conversion test passed")
        return True
        
    except ImportError:
        logger.warning("  ⚠ TensorFlow not available, skipping TFLite test")
        return True


def test_webcam_quick(duration: int = 5):
    """Quick webcam test (optional)."""
    logger.info(f"TEST 6: Webcam Quick Test ({duration}s)")
    
    import cv2
    import mediapipe as mp
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.warning("  ⚠ Webcam not available, skipping")
        return True
    
    mp_holistic = mp.solutions.holistic
    frames_processed = 0
    detections = 0
    
    start_time = time.time()
    
    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            frames_processed += 1
            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                detections += 1
            
            cv2.imshow('Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    fps = frames_processed / elapsed
    detect_rate = detections / max(frames_processed, 1)
    
    logger.info(f"  Frames: {frames_processed}, FPS: {fps:.1f}, Detection rate: {detect_rate:.1%}")
    logger.info("  ✓ Webcam test passed")
    return True


def run_all_tests(skip_webcam: bool = False):
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ISL RECOGNITION SYSTEM - TEST PIPELINE")
    print("=" * 60 + "\n")
    
    tests = [
        ("Landmark Extraction", test_landmark_extraction),
        ("Feature Aggregation", test_feature_aggregation),
        ("Synthetic Training", test_synthetic_training),
        ("Model Save/Load", test_model_save_load),
        ("TFLite Conversion", test_tflite_conversion),
    ]
    
    if not skip_webcam:
        tests.append(("Webcam Quick", lambda: test_webcam_quick(3)))
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            logger.error(f"  ✗ {name} FAILED: {e}")
            results.append((name, False, str(e)))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED\n")
    else:
        print("\n❌ SOME TESTS FAILED\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='ISL Recognition Test Pipeline')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'landmark', 'aggregation', 'training', 'tflite', 'webcam'])
    parser.add_argument('--skip-webcam', action='store_true', help='Skip webcam test')
    args = parser.parse_args()
    
    if args.test == 'all':
        run_all_tests(skip_webcam=args.skip_webcam)
    elif args.test == 'landmark':
        test_landmark_extraction()
    elif args.test == 'aggregation':
        test_feature_aggregation()
    elif args.test == 'training':
        test_synthetic_training()
    elif args.test == 'tflite':
        test_tflite_conversion()
    elif args.test == 'webcam':
        test_webcam_quick(30)


if __name__ == '__main__':
    main()
