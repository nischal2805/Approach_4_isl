#!/usr/bin/env python3
"""
ISL Recognition System - Model Training Module

This module handles:
1. Loading preprocessed features
2. Training RandomForest classifier
3. Evaluation with metrics and visualizations
4. Saving trained model and artifacts
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(input_folder: str) -> Tuple:
    """Load preprocessed training data."""
    input_path = Path(input_folder)
    logger.info(f"Loading data from: {input_folder}")
    
    X_train = np.load(input_path / 'X_train.npy')
    X_val = np.load(input_path / 'X_val.npy')
    X_test = np.load(input_path / 'X_test.npy')
    y_train = np.load(input_path / 'y_train.npy')
    y_val = np.load(input_path / 'y_val.npy')
    y_test = np.load(input_path / 'y_test.npy')
    label_encoder = joblib.load(input_path / 'label_encoder.pkl')
    scaler = joblib.load(input_path / 'scaler.pkl')
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = 200, max_depth: int = 20,
                        min_samples_split: int = 5) -> RandomForestClassifier:
    """Train RandomForest classifier."""
    logger.info(f"Training RF: n_estimators={n_estimators}, max_depth={max_depth}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, random_state=42,
        class_weight='balanced', n_jobs=-1, verbose=1
    )
    model.fit(X_train, y_train)
    logger.info("Training completed!")
    return model


def evaluate_model(model, X: np.ndarray, y: np.ndarray,
                   label_encoder: LabelEncoder, name: str = "Test") -> Dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    cm = confusion_matrix(y, y_pred)
    
    logger.info(f"{name}: Acc={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    print(f"\n{name} Classification Report:")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_))
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'confusion_matrix': cm, 'predictions': y_pred, 'probabilities': y_proba}


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_comparison(train_m: Dict, val_m: Dict, test_m: Dict, output_path: str):
    """Plot accuracy comparison."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, m) in enumerate([('Train', train_m), ('Val', val_m), ('Test', test_m)]):
        vals = [m[k] for k in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name)
        for bar, v in zip(bars, vals):
            ax.annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.1)
    ax.set_xticks(x + width); ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_model_artifacts(model, label_encoder, scaler, output_folder: str, metrics: Dict):
    """Save trained model and artifacts."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, output_path / 'model.pkl')
    joblib.dump(label_encoder, output_path / 'label_encoder.pkl')
    joblib.dump(scaler, output_path / 'scaler.pkl')
    
    with open(output_path / 'labels.txt', 'w') as f:
        for cls in label_encoder.classes_:
            f.write(f"{cls}\n")
    
    with open(output_path / 'metrics.txt', 'w') as f:
        f.write(f"ISL Recognition - {datetime.now()}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1: {metrics['f1']:.4f}\n")
    
    logger.info(f"Saved model to: {output_folder}")


def train_with_synthetic_data(num_classes: int = 10, samples_per_class: int = 20,
                              output_folder: str = 'models'):
    """Train with synthetic data for testing."""
    from data_preprocessing import create_synthetic_dataset
    from sklearn.model_selection import train_test_split
    
    X, y, class_names = create_synthetic_dataset(num_classes, samples_per_class)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    model = train_random_forest(X_train, y_train, n_estimators=50, max_depth=10)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    test_metrics = evaluate_model(model, X_test, y_test, label_encoder, "Test")
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(output_folder) / 'model.pkl')
    joblib.dump(label_encoder, Path(output_folder) / 'label_encoder.pkl')
    joblib.dump(scaler, Path(output_folder) / 'scaler.pkl')
    
    logger.info(f"Synthetic training done. Accuracy: {test_metrics['accuracy']:.4f}")
    return model, test_metrics['accuracy']


def main():
    parser = argparse.ArgumentParser(description='ISL Model Training')
    parser.add_argument('--input', type=str, default='data/processed')
    parser.add_argument('--output', type=str, default='models')
    parser.add_argument('--results', type=str, default='results')
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=20)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.results).mkdir(parents=True, exist_ok=True)
    
    if args.synthetic:
        train_with_synthetic_data(output_folder=args.output)
        return
    
    try:
        data = load_training_data(args.input)
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler = data
        
        model = train_random_forest(X_train, y_train, args.n_estimators, args.max_depth)
        
        train_m = evaluate_model(model, X_train, y_train, label_encoder, "Train")
        val_m = evaluate_model(model, X_val, y_val, label_encoder, "Val")
        test_m = evaluate_model(model, X_test, y_test, label_encoder, "Test")
        
        results_path = Path(args.results)
        plot_confusion_matrix(test_m['confusion_matrix'], label_encoder.classes_,
                            str(results_path / 'confusion_matrix.png'))
        plot_accuracy_comparison(train_m, val_m, test_m, str(results_path / 'accuracy.png'))
        
        save_model_artifacts(model, label_encoder, scaler, args.output, test_m)
        
        print(f"\n{'='*50}\nTRAINING COMPLETE - Test Accuracy: {test_m['accuracy']*100:.2f}%\n{'='*50}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
