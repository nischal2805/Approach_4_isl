"""
ISL Backend - Configuration
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
RECOGNITION_DIR = BASE_DIR / "isl_recognition"
TRANSLATION_DIR = BASE_DIR / "isl_translation"

# Model paths (Sign-to-Text)
MODEL_PATH = RECOGNITION_DIR / "models" / "model.pkl"
SCALER_PATH = RECOGNITION_DIR / "models" / "scaler.pkl"  
LABEL_ENCODER_PATH = RECOGNITION_DIR / "models" / "label_encoder.pkl"
LABELS_PATH = RECOGNITION_DIR / "models" / "labels.txt"

# Text-to-Sign frames path
FRAMES_WORD_LEVEL = TRANSLATION_DIR / "data" / "isl_clstr" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus" / "Frames_Word_Level"

# MediaPipe settings
LANDMARKS_PER_FRAME = 225  # 75 keypoints * 3 coords (pose=33, left_hand=21, right_hand=21)
FEATURE_DIM = 675  # 225 * 3 (mean, max, std)
BUFFER_SIZE = 30  # frames per sign (1 second at 30fps)

# Server settings
HOST = "0.0.0.0"
PORT = 8000
