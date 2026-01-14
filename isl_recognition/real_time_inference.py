#!/usr/bin/env python3
"""
ISL Recognition System - Real-Time Inference Module

This module provides real-time sign language recognition using webcam feed
with sliding window landmark extraction and smoothed predictions.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from collections import deque
from typing import Optional, List, Tuple

import cv2
import numpy as np
import mediapipe as mp
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Constants
LANDMARKS_PER_FRAME = 225  # 75 keypoints * 3 coords
BUFFER_SIZE = 30  # 1 second at 30 FPS
PREDICT_INTERVAL = 10  # Predict every N frames
SMOOTHING_WINDOW = 3  # Majority vote over last N predictions


class RealTimeRecognizer:
    """Real-time ISL recognition using webcam."""
    
    def __init__(self, model_path: str, label_encoder_path: str, scaler_path: str):
        """Initialize recognizer with trained model."""
        logger.info("Loading model artifacts...")
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.scaler = joblib.load(scaler_path)
        
        self.class_names = list(self.label_encoder.classes_)
        logger.info(f"Loaded {len(self.class_names)} classes")
        
        # Buffers
        self.landmark_buffer = deque(maxlen=BUFFER_SIZE)
        self.prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
        
        # State
        self.frame_count = 0
        self.current_prediction = "Waiting..."
        self.current_confidence = 0.0
        self.fps = 0.0
        
    def extract_frame_landmarks(self, results) -> Optional[np.ndarray]:
        """Extract landmarks from MediaPipe results."""
        landmarks = []
        
        # Pose (33 * 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)
        
        # Left hand (21 * 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Right hand (21 * 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        return np.array(landmarks, dtype=np.float32)
    
    def aggregate_features(self, landmarks: np.ndarray) -> np.ndarray:
        """Aggregate temporal features (mean, max, std)."""
        mean_f = np.mean(landmarks, axis=0)
        max_f = np.max(landmarks, axis=0)
        std_f = np.std(landmarks, axis=0)
        return np.concatenate([mean_f, max_f, std_f])
    
    def predict(self) -> Tuple[str, float]:
        """Run prediction on buffered landmarks."""
        if len(self.landmark_buffer) < 10:
            return "Collecting...", 0.0
        
        landmarks = np.array(self.landmark_buffer)
        features = self.aggregate_features(landmarks)
        features = self.scaler.transform(features.reshape(1, -1))
        
        proba = self.model.predict_proba(features)[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        
        self.prediction_buffer.append(pred_idx)
        
        # Majority vote
        if len(self.prediction_buffer) >= 2:
            votes = list(self.prediction_buffer)
            final_pred = max(set(votes), key=votes.count)
        else:
            final_pred = pred_idx
        
        return self.class_names[final_pred], confidence
    
    def draw_landmarks(self, frame, results):
        """Draw MediaPipe landmarks on frame."""
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2))
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2))
    
    def draw_info(self, frame):
        """Draw prediction info on frame."""
        h, w = frame.shape[:2]
        
        # Background box
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 255, 0), 2)
        
        # Text
        cv2.putText(frame, f"Sign: {self.current_prediction}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {self.current_confidence:.1%}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (w - 180, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def run(self, camera_index: int = 0):
        """Run real-time recognition loop."""
        logger.info(f"Starting webcam (index: {camera_index})...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        prev_time = time.time()
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            logger.info("Recognition started. Press 'q' to quit.")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                
                # Extract and buffer landmarks
                landmarks = self.extract_frame_landmarks(results)
                self.landmark_buffer.append(landmarks)
                self.frame_count += 1
                
                # Predict at interval
                if self.frame_count % PREDICT_INTERVAL == 0:
                    self.current_prediction, self.current_confidence = self.predict()
                
                # Calculate FPS
                curr_time = time.time()
                self.fps = 1.0 / (curr_time - prev_time + 1e-8)
                prev_time = curr_time
                
                # Draw
                self.draw_landmarks(frame, results)
                self.draw_info(frame)
                
                cv2.imshow('ISL Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Recognition stopped.")


def main():
    parser = argparse.ArgumentParser(description='ISL Real-Time Recognition')
    parser.add_argument('--model', type=str, default='models/model.pkl')
    parser.add_argument('--encoder', type=str, default='models/label_encoder.pkl')
    parser.add_argument('--scaler', type=str, default='models/scaler.pkl')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    
    # Validate files exist
    for path in [args.model, args.encoder, args.scaler]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            logger.error("Please train the model first using train_model.py")
            sys.exit(1)
    
    recognizer = RealTimeRecognizer(args.model, args.encoder, args.scaler)
    recognizer.run(args.camera)


if __name__ == '__main__':
    main()
