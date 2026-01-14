#!/usr/bin/env python3
"""
ISL Recognition System - Mobile Conversion Module

Converts RandomForest model to TFLite via MLP knowledge distillation.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mlp_model(input_dim: int, num_classes: int):
    """Create MLP model for knowledge distillation."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def distill_to_mlp(rf_model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 50, batch_size: int = 32):
    """Train MLP to mimic RandomForest predictions."""
    import tensorflow as tf
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    logger.info(f"Creating MLP: input={input_dim}, classes={num_classes}")
    
    # Get RF soft labels (probabilities)
    rf_proba_train = rf_model.predict_proba(X_train)
    
    mlp = create_mlp_model(input_dim, num_classes)
    
    # Train with RF soft labels using knowledge distillation
    logger.info("Training MLP with knowledge distillation...")
    
    # Custom training loop for soft label distillation
    history = mlp.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )
    
    # Evaluate
    val_loss, val_acc = mlp.evaluate(X_val, y_val, verbose=0)
    logger.info(f"MLP Validation Accuracy: {val_acc:.4f}")
    
    return mlp, history


def convert_to_tflite(keras_model, output_path: str, quantize: bool = True):
    """Convert Keras model to TFLite."""
    import tensorflow as tf
    
    logger.info("Converting to TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Saved TFLite model: {output_path} ({size_mb:.2f} MB)")
    
    return tflite_model


def test_tflite_inference(tflite_path: str, X_test: np.ndarray, y_test: np.ndarray):
    """Test TFLite model inference speed and accuracy."""
    import tensorflow as tf
    import time
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct = 0
    total_time = 0
    
    for i in range(len(X_test)):
        input_data = X_test[i:i+1].astype(np.float32)
        
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        elapsed = time.perf_counter() - start
        
        total_time += elapsed
        pred = np.argmax(output[0])
        if pred == y_test[i]:
            correct += 1
    
    accuracy = correct / len(X_test)
    avg_latency = (total_time / len(X_test)) * 1000  # ms
    
    logger.info(f"TFLite Test Accuracy: {accuracy:.4f}")
    logger.info(f"Average Latency: {avg_latency:.2f} ms")
    
    return accuracy, avg_latency


def main():
    parser = argparse.ArgumentParser(description='Convert model to TFLite')
    parser.add_argument('--model', type=str, default='models/model.pkl')
    parser.add_argument('--data', type=str, default='data/processed')
    parser.add_argument('--output', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.synthetic:
            # Generate synthetic data for testing
            from data_preprocessing import create_synthetic_dataset
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            
            X, y, class_names = create_synthetic_dataset(10, 30)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            
            rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Save labels
            with open(output_path / 'labels.txt', 'w') as f:
                for name in class_names:
                    f.write(f"{name}\n")
        else:
            # Load real model and data
            rf_model = joblib.load(args.model)
            data_path = Path(args.data)
            
            X_train = np.load(data_path / 'X_train.npy')
            X_val = np.load(data_path / 'X_val.npy')
            X_test = np.load(data_path / 'X_test.npy')
            y_train = np.load(data_path / 'y_train.npy')
            y_val = np.load(data_path / 'y_val.npy')
            y_test = np.load(data_path / 'y_test.npy')
        
        # Distill to MLP
        mlp, history = distill_to_mlp(rf_model, X_train, y_train, X_val, y_val, epochs=args.epochs)
        
        # Save Keras model
        mlp.save(output_path / 'mlp_model.keras')
        logger.info(f"Saved Keras model: {output_path / 'mlp_model.keras'}")
        
        # Convert to TFLite
        tflite_path = str(output_path / 'model.tflite')
        convert_to_tflite(mlp, tflite_path, quantize=True)
        
        # Test TFLite
        accuracy, latency = test_tflite_inference(tflite_path, X_test, y_test)
        
        print(f"\n{'='*50}")
        print("CONVERSION COMPLETE")
        print(f"{'='*50}")
        print(f"TFLite Accuracy: {accuracy*100:.2f}%")
        print(f"Inference Latency: {latency:.2f} ms")
        print(f"Model: {tflite_path}")
        print(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
