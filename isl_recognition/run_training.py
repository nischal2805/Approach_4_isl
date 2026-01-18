#!/usr/bin/env python3
"""
ISL Recognition - Complete Training Pipeline Runner

This script runs the entire training pipeline:
1. Download dataset (optional)
2. Preprocess videos -> extract features
3. Add NO_SIGN class
4. Train RandomForest classifier
5. Export to TFLite

Usage:
    # Full pipeline
    python run_training.py --dataset data/INCLUDE50 --output models
    
    # Skip preprocessing (use existing processed data)
    python run_training.py --processed data/processed --output models
"""

import os
import sys
import argparse
import time
from pathlib import Path


def run_preprocessing(dataset_dir: str, output_dir: str):
    """Run data preprocessing."""
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing Dataset")
    print("=" * 60)
    
    from data_preprocessing import process_dataset
    
    start_time = time.time()
    
    X_train, X_val, X_test, y_train, y_val, y_test = process_dataset(
        dataset_dir,
        output_dir,
        test_size=0.15,
        val_size=0.15
    )
    
    elapsed = time.time() - start_time
    print(f"\nPreprocessing completed in {elapsed/60:.1f} minutes")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return output_dir


def run_add_no_sign(processed_dir: str, count: int = 200):
    """Add NO_SIGN class to processed data."""
    print("\n" + "=" * 60)
    print("STEP 2: Adding NO_SIGN Class")
    print("=" * 60)
    
    from generate_no_sign import add_no_sign_to_processed_data
    
    add_no_sign_to_processed_data(processed_dir, count)


def run_training(processed_dir: str, models_dir: str, results_dir: str):
    """Run model training."""
    print("\n" + "=" * 60)
    print("STEP 3: Training RandomForest Classifier")
    print("=" * 60)
    
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    start_time = time.time()
    
    # Load data
    processed_path = Path(processed_dir)
    X_train = np.load(processed_path / 'X_train.npy')
    X_val = np.load(processed_path / 'X_val.npy')
    X_test = np.load(processed_path / 'X_test.npy')
    y_train = np.load(processed_path / 'y_train.npy')
    y_val = np.load(processed_path / 'y_val.npy')
    y_test = np.load(processed_path / 'y_test.npy')
    label_encoder = joblib.load(processed_path / 'label_encoder.pkl')
    
    print(f"Training data: {X_train.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Train RandomForest
    print("\nTraining RandomForest (this may take a few minutes)...")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy:   {val_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Test F1 Score:  {test_f1:.4f}")
    
    # Save model
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, models_path / 'model.pkl')
    
    # Copy label encoder and scaler
    import shutil
    shutil.copy(processed_path / 'label_encoder.pkl', models_path / 'label_encoder.pkl')
    shutil.copy(processed_path / 'scaler.pkl', models_path / 'scaler.pkl')
    shutil.copy(processed_path / 'labels.txt', models_path / 'labels.txt')
    
    # Save metrics
    with open(models_path / 'metrics.txt', 'w') as f:
        f.write(f"ISL Recognition - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Classes: {len(label_encoder.classes_)}\n")
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Val Accuracy: {val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
    
    # Save classification report
    if results_dir:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        report = classification_report(
            y_test, test_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        import pandas as pd
        pd.DataFrame(report).transpose().to_csv(results_path / 'classification_report.csv')
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Model saved to: {models_path / 'model.pkl'}")
    
    return models_path


def run_tflite_export(models_dir: str, processed_dir: str):
    """Export model to TFLite format."""
    print("\n" + "=" * 60)
    print("STEP 4: Exporting to TFLite")
    print("=" * 60)
    
    import subprocess
    
    cmd = [
        sys.executable, 'convert_to_mobile.py',
        '--model', str(Path(models_dir) / 'model.pkl'),
        '--data', processed_dir,
        '--output', models_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: TFLite export may have failed")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    else:
        print(result.stdout)
        print(f"TFLite model saved to: {models_dir}/model.tflite")


def run_scaler_export(processed_dir: str, app_assets_dir: str):
    """Export scaler to JSON for Flutter app."""
    print("\n" + "=" * 60)
    print("STEP 5: Exporting Scaler to JSON")
    print("=" * 60)
    
    import json
    from pathlib import Path
    import joblib
    
    input_path = Path(processed_dir)
    output_path = Path(app_assets_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        scaler = joblib.load(input_path / 'scaler.pkl')
        scaler_data = {
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist()
        }
        
        output_file = output_path / 'scaler.json'
        with open(output_file, 'w') as f:
            json.dump(scaler_data, f)
        
        print(f"Scaler exported to: {output_file}")
    except Exception as e:
        print(f"Warning: Failed to export scaler: {e}")


def main():
    parser = argparse.ArgumentParser(description='ISL Recognition Training Pipeline')
    
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to raw dataset (e.g., data/INCLUDE50)')
    parser.add_argument('--processed', type=str, default=None,
                       help='Path to already processed data (skip preprocessing)')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--results', type=str, default='results',
                       help='Output directory for results/plots')
    parser.add_argument('--no-sign-count', type=int, default=200,
                       help='Number of NO_SIGN samples to add')
    parser.add_argument('--skip-tflite', action='store_true',
                       help='Skip TFLite export')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.processed is None and args.dataset is None:
        print("Error: Must specify either --dataset or --processed")
        sys.exit(1)
    
    processed_dir = args.processed
    
    # Step 1: Preprocess if needed
    if processed_dir is None:
        processed_dir = 'data/processed'
        run_preprocessing(args.dataset, processed_dir)
    
    # Step 2: Add NO_SIGN class
    run_add_no_sign(processed_dir, args.no_sign_count)
    
    # Step 3: Train model
    models_path = run_training(processed_dir, args.output, args.results)
    
    # Step 4: Export to TFLite
    if not args.skip_tflite:
        run_tflite_export(args.output, processed_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Processed data: {processed_dir}")
    print(f"  Models: {args.output}")
    print(f"  Results: {args.results}")
    print(f"\nNext steps:")
    print(f"  1. Test with: python real_time_inference.py --camera 0")
    print(f"  2. Copy to Flutter: cp {args.output}/model.tflite ../isl_app/assets/")


if __name__ == '__main__':
    main()
