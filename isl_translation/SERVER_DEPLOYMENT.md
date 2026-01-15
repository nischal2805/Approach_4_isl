# Server Deployment Guide - A100 Training

## Pre-requisites on Server

```bash
# Python 3.8+
python --version

# CUDA 11.8+ with cuDNN
nvidia-smi

# Git
git --version
```

## Setup Steps

### 1. Clone Repository
```bash
git clone https://github.com/nischal2805/Approach_4_isl.git
cd Approach_4_isl/isl_translation
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download I3D Pretrained Weights
```bash
# Option 1: wget
wget https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt -O models/rgb_imagenet.pt

# Option 2: Manual
# Download from: https://github.com/piergiaj/pytorch-i3d/tree/master/models
# Place in: isl_translation/models/rgb_imagenet.pt
```

### 5. Verify Dataset Exists
```bash
python verify_deployment.py
# Should show: ✓ ISL-CSLTR Videos: ... 
# Total videos: 1374
```

## Training on A100

### Full Training (Recommended)
```bash
# 50 epochs, batch_size=8, mixed precision
python training/trainer.py --config configs/train_config.yaml --device cuda
```

**Expected:**
- Training time: 24-36 hours
- BLEU score: 20-30
- Checkpoint saved: `checkpoints/best_model.pt`

### Monitor Training
```bash
# In another terminal
watch -n 30 tail -100 training.log

# Or use grep for summaries
grep "Epoch" training.log
```

### Resume from Checkpoint (if interrupted)
```bash
python training/trainer.py \
    --config configs/train_config.yaml \
    --resume checkpoints/checkpoint_epoch_XXX.pt \
    --device cuda
```

## After Training

### Evaluate Model
```bash
python training/trainer.py \
    --config configs/train_config.yaml \
    --evaluate-only \
    --device cuda
```

### Test Inference
```bash
python test_inference.py
```

### Plot Training Curves
```bash
python plot_metrics.py
# Saves to: logs/training_metrics.png
```

## Deploy API Server

### Local Testing
```bash
python inference/api.py
# Server at http://localhost:8000
```

### Production (AWS Lambda)
```bash
# Install deployment dependencies
pip install mangum

# Package for Lambda
# (Requires separate Lambda deployment guide)
```

## Troubleshooting

### Out of Memory
```yaml
# Edit configs/train_config.yaml
training:
  batch_size: 4  # Reduce from 8
  accumulation_steps: 8  # Increase to keep effective batch
```

### NaN Losses
```yaml
stability:
  mixed_precision: false  # Disable AMP
  gradient_clip: 0.5      # More aggressive clipping
  learning_rate: 3.0e-5   # Lower LR
```

### Slow Data Loading
```yaml
data:
  num_workers: 8  # Increase (adjust to CPU cores)
```

## Expected Files After Training

```
checkpoints/
├── best_model.pt              # Best validation loss
├── checkpoint_epoch_050.pt    # Final epoch
└── ...

logs/
├── metrics.json               # Training history
└── training_metrics.png       # Loss/BLEU plots

training.log                   # Full training log
```

## Download Trained Model to Local

```bash
# From server
scp user@server:~/Approach_4_isl/isl_translation/checkpoints/best_model.pt ./

# Then test locally
python test_inference.py
```

## Key Monitoring Metrics

| Metric | Good | Acceptable | Bad |
|--------|------|------------|-----|
| Train Loss | <1.0 | 1.0-2.0 | >3.0 |
| Val Loss | <1.5 | 1.5-2.5 | >3.0 |
| Val BLEU | >25 | 15-25 | <10 |
| Time/Epoch | <30min | 30-60min | >60min |
