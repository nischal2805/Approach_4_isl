# ISL Recognition System

Real-time Indian Sign Language (ISL) recognition using MediaPipe landmarks and RandomForest classifier with mobile deployment via TFLite.

## Features

- **MediaPipe Holistic**: Extracts 75 keypoints (pose + hands) per frame
- **Temporal Aggregation**: Mean, max, std features across video → 675-dimensional vector
- **RandomForest Classifier**: Fast training, good accuracy (85%+ target)
- **Real-time Webcam**: 25-30 FPS inference with sliding window
- **Mobile Deployment**: TFLite conversion via MLP knowledge distillation

## Quick Start

### 1. Setup Environment

```bash
# Using existing venv at E:\5thsem el\kortex_5th_sem\kortex
# Or install dependencies:
pip install -r requirements.txt
```

### 2. Download Dataset

1. Go to [INCLUDE-50 Dataset](https://zenodo.org/record/4010759)
2. Download and extract to `data/INCLUDE50/`
3. Structure should be: `data/INCLUDE50/<class_name>/<video>.mp4`

### 3. Test Pipeline (Synthetic Data)

```bash
# Run all tests without dataset
python test_pipeline.py --skip-webcam
```

### 4. Process Dataset

```bash
python data_preprocessing.py --input data/INCLUDE50 --output data/processed
```

### 5. Train Model

```bash
python train_model.py --input data/processed --output models --results results
```

### 6. Real-Time Demo

```bash
python real_time_inference.py --camera 0
```

### 7. Convert to Mobile

```bash
python convert_to_mobile.py --model models/model.pkl --data data/processed --output models
```

## Project Structure

```
isl_recognition/
├── data_preprocessing.py   # Landmark extraction & feature aggregation
├── train_model.py          # RandomForest training & evaluation
├── real_time_inference.py  # Webcam-based recognition
├── convert_to_mobile.py    # TFLite conversion
├── mobile_demo_template.py # Android integration code
├── test_pipeline.py        # End-to-end tests
├── requirements.txt
├── README.md
├── data/                   # Dataset folder
│   └── processed/          # Preprocessed features
├── models/                 # Saved models
│   ├── model.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── model.tflite
└── results/                # Plots & metrics
```

## Performance Targets

| Metric | Target | 
|--------|--------|
| Test Accuracy | 85%+ |
| Real-time FPS | 25-30 |
| Model Size | <50MB |
| Inference Latency | <50ms |
| Training Time | <30 min |

## Architecture

```
Video Frame
     ↓
MediaPipe Holistic → 21 left + 21 right + 33 pose = 75 keypoints
     ↓
Extract (x, y, z) per keypoint → 225 features/frame
     ↓
Aggregate across frames → mean + max + std = 675 features/video
     ↓
StandardScaler → Normalize features
     ↓
RandomForest (200 trees, depth=20) → Prediction + Confidence
```

## Troubleshooting

**MediaPipe not detecting hands:**
- Ensure good lighting
- Keep hands in frame
- Face camera directly

**Low accuracy:**
- Increase `n_estimators` (more trees)
- Check data quality
- Verify class balance

**Slow inference:**
- Reduce `n_estimators`
- Use TFLite model
- Lower camera resolution

**TFLite conversion fails:**
- Install TensorFlow 2.15+
- Check Keras model saves correctly

## Mobile Integration

See `mobile_demo_template.py` for:
- Python TFLite inference
- Android Kotlin integration
- MediaPipe Android SDK usage

Required Android assets:
- `model.tflite` (from models/)
- `labels.txt` (class names)
- `holistic_landmarker.task` (MediaPipe)

## License

MIT License
