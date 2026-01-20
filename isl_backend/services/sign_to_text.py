"""
ISL Backend - Sign-to-Text Service

Uses MediaPipe Holistic for landmark extraction and trained model for classification.
This matches EXACTLY how the training data was processed.
"""
import cv2
import numpy as np
import mediapipe as mp
import joblib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

from config import (
    MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH,
    LANDMARKS_PER_FRAME, FEATURE_DIM, BUFFER_SIZE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipe setup - EXACTLY as used in training
mp_holistic = mp.solutions.holistic


class SignToTextService:
    """Service for recognizing ISL signs from video frames."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.holistic = None
        self._loaded = False
        
    def load(self) -> bool:
        """Load model and initialize MediaPipe."""
        try:
            logger.info("Loading Sign-to-Text model...")
            
            # Load trained model
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            
            # Initialize MediaPipe Holistic
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self._loaded = True
            logger.info(f"Loaded model with {len(self.label_encoder.classes_)} classes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def class_names(self) -> List[str]:
        if self.label_encoder is None:
            return []
        return list(self.label_encoder.classes_)
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract landmarks from a single frame using MediaPipe Holistic.
        Returns 225-dimensional vector (75 keypoints * 3 coords).
        
        This matches EXACTLY how training data was processed!
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        landmarks = []
        
        # Pose landmarks (33 keypoints * 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)
        
        # Left hand landmarks (21 keypoints * 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Right hand landmarks (21 keypoints * 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        return np.array(landmarks, dtype=np.float32)
    
    def aggregate_features(self, frames_landmarks: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate frame landmarks into features: [mean, max, std].
        Input: List of 225-dim arrays -> Output: 675-dim array
        
        This matches EXACTLY how training features were created!
        """
        if not frames_landmarks:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        
        # Stack all frames: (num_frames, 225)
        stacked = np.stack(frames_landmarks)
        
        # Compute statistics along time axis
        mean_features = np.mean(stacked, axis=0)  # (225,)
        max_features = np.max(stacked, axis=0)    # (225,)
        std_features = np.std(stacked, axis=0)    # (225,)
        
        # Concatenate: [mean, max, std] = 675 features
        features = np.concatenate([mean_features, max_features, std_features])
        
        return features.astype(np.float32)
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Run prediction on aggregated features.
        Returns: (label, confidence, top_5_predictions)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        # Scale features using the same scaler from training
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and probabilities
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get label
        label = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_5 = {
            self.label_encoder.inverse_transform([idx])[0]: float(probabilities[idx])
            for idx in top_indices
        }
        
        return label, confidence, top_5
    
    def process_video_bytes(self, video_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Process a video file and return detected signs.
        
        Args:
            video_bytes: Raw video file bytes
            
        Returns:
            List of detected signs with timestamps and confidence
        """
        import tempfile
        import os
        
        # Write bytes to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name
        
        try:
            return self.process_video_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def process_video_file(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Process a video file and return detected signs.
        Uses sliding window of 30 frames with 15-frame step.
        """
        if not self._loaded:
            self.load()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {frame_count} frames at {fps} FPS")
        
        # Extract all frame landmarks
        all_landmarks = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = self.extract_landmarks(frame)
            all_landmarks.append(landmarks)
        
        cap.release()
        
        if len(all_landmarks) < BUFFER_SIZE:
            logger.warning(f"Video too short: {len(all_landmarks)} frames")
            # Process whatever we have
            if all_landmarks:
                features = self.aggregate_features(all_landmarks)
                label, conf, top5 = self.predict(features)
                return [{
                    "sign": label,
                    "confidence": conf,
                    "start_frame": 0,
                    "end_frame": len(all_landmarks),
                    "start_time": 0.0,
                    "end_time": len(all_landmarks) / fps,
                    "top_predictions": top5
                }]
            return []
        
        # Sliding window detection
        window_size = BUFFER_SIZE
        step_size = window_size // 2  # 50% overlap
        
        results = []
        for start in range(0, len(all_landmarks) - window_size + 1, step_size):
            end = start + window_size
            window_landmarks = all_landmarks[start:end]
            
            features = self.aggregate_features(window_landmarks)
            label, conf, top5 = self.predict(features)
            
            # Only include if confidence is reasonable
            if conf >= 0.15:  # 15% threshold
                results.append({
                    "sign": label,
                    "confidence": conf,
                    "start_frame": start,
                    "end_frame": end,
                    "start_time": start / fps,
                    "end_time": end / fps,
                    "top_predictions": top5
                })
        
        # Merge consecutive same predictions
        merged = self._merge_consecutive(results)
        
        return merged
    
    def _merge_consecutive(self, results: List[Dict]) -> List[Dict]:
        """Merge consecutive windows with the same prediction."""
        if not results:
            return []
        
        merged = [results[0].copy()]
        
        for r in results[1:]:
            if r["sign"] == merged[-1]["sign"]:
                # Extend the previous result
                merged[-1]["end_frame"] = r["end_frame"]
                merged[-1]["end_time"] = r["end_time"]
                merged[-1]["confidence"] = max(merged[-1]["confidence"], r["confidence"])
            else:
                merged.append(r.copy())
        
        return merged
    
    def process_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Process a list of frames (as numpy arrays) and return prediction.
        For real-time streaming use case.
        """
        if not self._loaded:
            self.load()
        
        # Extract landmarks from all frames
        landmarks_list = []
        for frame in frames:
            lm = self.extract_landmarks(frame)
            landmarks_list.append(lm)
        
        # Aggregate and predict
        features = self.aggregate_features(landmarks_list)
        label, conf, top5 = self.predict(features)
        
        return {
            "sign": label,
            "confidence": conf,
            "num_frames": len(frames),
            "top_predictions": top5
        }
    
    def cleanup(self):
        """Release MediaPipe resources."""
        if self.holistic:
            self.holistic.close()


# Singleton instance
_service: Optional[SignToTextService] = None

def get_sign_to_text_service() -> SignToTextService:
    global _service
    if _service is None:
        _service = SignToTextService()
        _service.load()
    return _service
