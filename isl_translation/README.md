# ISL Sentence Translation System

End-to-end Indian Sign Language to English translation using I3D + BiLSTM + T5.

## Quick Start

### 1. Download Dataset
Download ISL-CSLTR from Kaggle and extract to `data/isl_csltr/`

### 2. Download I3D Pretrained Weights
```bash
# Download from: https://github.com/piergiaj/pytorch-i3d
wget https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt
mv rgb_imagenet.pt models/
```

### 3. Train
```bash
# On A100 (recommended)
python training/trainer.py --config configs/train_config.yaml --device cuda

# On CPU (slow, for testing only)
python training/trainer.py --config configs/train_config.yaml --device cpu
```

### 4. Evaluate
```bash
python training/trainer.py --config configs/train_config.yaml --evaluate-only
```

## Project Structure

```
isl_translation/
├── configs/
│   └── train_config.yaml      # Training hyperparameters
├── data/
│   ├── dataset.py             # Dataset loader
│   └── isl_csltr/             # Dataset goes here
├── models/
│   ├── i3d_encoder.py         # I3D video feature extractor
│   ├── temporal_encoder.py    # BiLSTM + projection
│   └── translator.py          # Full seq2seq model
├── training/
│   ├── trainer.py             # Training loop
│   └── utils.py               # Stability utilities
├── checkpoints/               # Saved models
└── logs/                      # Training logs
```

## Architecture

```
Video [B, 3, T, H, W]
    ↓
I3D Encoder (frozen, pretrained)
    ↓
[B, T', 1024] frame features
    ↓
BiLSTM (2 layers, 512 hidden)
    ↓
Projection + Positional Encoding
    ↓
[B, T', 512]
    ↓
T5-Small Decoder
    ↓
English sentence
```

## Training Features

- ✅ Mixed precision (FP16)
- ✅ Gradient clipping (max_norm=1.0)
- ✅ NaN/Inf protection
- ✅ Warmup + cosine LR schedule
- ✅ Gradient accumulation
- ✅ Early stopping
- ✅ Checkpointing
- ✅ BLEU evaluation

## Expected Results

| Metric | Target |
|--------|--------|
| Test BLEU | 20-30 |
| Inference | 2-3 sec/video |
| Training | 24-36h on A100 |
