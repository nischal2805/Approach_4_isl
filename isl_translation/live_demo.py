#!/usr/bin/env python3
"""
Live Real-Time ISL Translation Demo.
Uses webcam to capture video and translates to English.
"""

import cv2
import torch
import numpy as np
import time
from pathlib import Path
from collections import deque
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models import ISLTranslator


class LiveTranslator:
    """Real-time ISL to English translator using webcam."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self._load_model(checkpoint_path)
        
        # Video settings
        self.num_frames = 30
        self.frame_size = 224
        self.fps = 15
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=self.num_frames)
        
        # Translation state
        self.last_translation = ""
        self.translation_history = []
        self.recording = False
        self.record_start_time = None
        
    def _load_model(self, checkpoint_path: str):
        """Load trained model."""
        print(f"Loading model from {checkpoint_path}...")
        
        model = ISLTranslator(
            t5_model_name='t5-small',
            freeze_i3d=True,
            lstm_hidden=512,
            lstm_layers=2
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully!")
        return model
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame."""
        frame = cv2.resize(frame, (self.frame_size, self.frame_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def prepare_video_tensor(self):
        """Convert frame buffer to model input tensor."""
        if len(self.frame_buffer) < self.num_frames:
            return None
        
        frames = list(self.frame_buffer)
        video = np.stack(frames, axis=0)  # [T, H, W, 3]
        video = video.transpose(3, 0, 1, 2)  # [3, T, H, W]
        video = video.astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
        video = (video - mean) / std
        
        return torch.from_numpy(video).unsqueeze(0).float().to(self.device)
    
    def translate(self):
        """Translate current frame buffer."""
        video_tensor = self.prepare_video_tensor()
        if video_tensor is None:
            return None
        
        with torch.no_grad():
            translation = self.model.translate(video_tensor, num_beams=4, max_length=50)[0]
        
        return translation
    
    def run(self, camera_id: int = 0):
        """Run live translation demo."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\n" + "="*50)
        print("ISL LIVE TRANSLATOR")
        print("="*50)
        print("Controls:")
        print("  SPACE - Start/Stop recording & translate")
        print("  R     - Reset translation")
        print("  Q     - Quit")
        print("="*50 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Preprocess and add to buffer if recording
            if self.recording:
                processed = self.preprocess_frame(frame)
                self.frame_buffer.append(processed)
                
                # Show recording indicator
                elapsed = time.time() - self.record_start_time
                cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(display_frame, f"Recording: {elapsed:.1f}s", 
                           (55, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Frames: {len(self.frame_buffer)}/{self.num_frames}",
                           (55, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show last translation
            if self.last_translation:
                cv2.rectangle(display_frame, (10, display_frame.shape[0]-80), 
                             (display_frame.shape[1]-10, display_frame.shape[0]-10), (0, 0, 0), -1)
                cv2.putText(display_frame, f"Translation: {self.last_translation}",
                           (20, display_frame.shape[0]-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(display_frame, "Press SPACE to record, Q to quit",
                       (10, display_frame.shape[0]-90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('ISL Live Translator', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - toggle recording
                if not self.recording:
                    self.recording = True
                    self.record_start_time = time.time()
                    self.frame_buffer.clear()
                    print("Recording started...")
                else:
                    self.recording = False
                    print("Recording stopped. Translating...")
                    
                    if len(self.frame_buffer) >= 10:
                        translation = self.translate()
                        self.last_translation = translation or "Could not translate"
                        self.translation_history.append(self.last_translation)
                        print(f"Translation: {self.last_translation}")
                    else:
                        print("Not enough frames recorded")
            
            elif key == ord('r'):  # Reset
                self.last_translation = ""
                self.frame_buffer.clear()
                print("Reset")
            
            elif key == ord('q'):  # Quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print history
        if self.translation_history:
            print("\n" + "="*50)
            print("TRANSLATION HISTORY")
            print("="*50)
            for i, t in enumerate(self.translation_history, 1):
                print(f"{i}. {t}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/server_best_model.pt')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    translator = LiveTranslator(args.checkpoint, args.device)
    translator.run(args.camera)


if __name__ == '__main__':
    main()
