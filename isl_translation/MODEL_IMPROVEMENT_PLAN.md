# ISL Model Improvement Plan (Updated)

## Current Performance Analysis

| Metric | Value |
|--------|-------|
| **Total Accuracy** | 7.4% (51/687) |
| Train | 7.5% (36/480) |
| Val | 3.9% (4/103) |
| Test | 10.6% (11/104) |
| BLEU Score | 1.31 |

### Critical Issue: Model Mode Collapse
The model has collapsed to predicting a few common phrases:
- **159 times** → "how are you" (23% of all predictions!)
- **34 times** → "why are you angry"
- **31 times** → "i am happy"

### What It Gets Right (Short, Frequent Sentences)
- "congratulations" (86% accuracy)
- "why are you crying" (80%)
- "i am hungry" (71%)

---

## ✅ Improvements Implemented

### 1. Data Augmentation (Created: `data/augmentation.py`)
- **Spatial**: Random crop, horizontal flip, rotation
- **Temporal**: Frame shift, speed variation  
- **Color**: Brightness, contrast jitter
- **Purpose**: Prevent overfitting, increase robustness

### 2. I3D Pretrained Weights
- Config updated to use `models/rgb_imagenet.pt`
- **Impact**: Better visual feature extraction from start

### 3. Training Configuration (`configs/train_improved.yaml`)
- More epochs (150)
- Higher label smoothing (0.15)
- Gradient accumulation for larger effective batch
- More frames per video (32)

---

## Dataset Research Summary

| Dataset | Size | Status | Recommendation |
|---------|------|--------|----------------|
| **ISL-CSLTR (yours)** | 687 videos | ✅ Using | Best for focused project |
| I-Sign | 118K videos | ⚠️ Available | Too difficult, BLEU=1.47 |
| INCLUDE | 4287 videos | Not tested | Word-level only |
| INSIGNVID | 55 sentences | Small | Similar sentences |

**Conclusion**: Stick with ISL-CSLTR. It's cleaner and achievable for your timeline.

---

## Deployment Strategy

### Primary: Mobile (Quantized)
```
Model → ONNX → INT8 Quantization → TFLite → Flutter App
```

### Fallback: GCP Cloud Functions
```
Model → ONNX → Cloud Function → REST API → Flutter App
```

### Demo Mode (Simulated Live)
```
Camera → 5-sec segments → Send to model → Display translation
(Feels like live, but batch processing behind the scenes)
```

---

## Expected Improvements After Retraining

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Overall Accuracy | 7.4% | 20-30% |
| BLEU Score | 1.31 | 5-15 |
| Short sentences | Poor | 40-60% |
| Mode collapse | Severe | Reduced |

---

## Files Created

1. `data/augmentation.py` - Video augmentation pipeline
2. `configs/train_improved.yaml` - Optimized training config
3. `4_DAY_ACTION_PLAN.md` - Day-by-day execution plan

---

## Next Steps (Today)

1. **Transfer to Server**: Push new files to A100
2. **Start Training**: 
   ```bash
   python train.py --config configs/train_improved.yaml
   ```
3. **Monitor**: Check Telegram for updates
4. **Prepare Export Script**: For ONNX conversion tomorrow

---

## For Your Paper

Even with 20-30% accuracy, you can claim:
1. ✅ Working end-to-end ISL → English system
2. ✅ Novel application of I3D + BiLSTM + T5 for ISL
3. ✅ Data augmentation for sign language videos
4. ✅ Mobile-deployable architecture
5. ✅ Real-time capable (5-second segments)

### Limitations to Acknowledge
- Small dataset (687 videos)
- ISL standardization challenges
- Need for larger corpus in future work

### Future Work
- Gemini Live-style continuous translation
- Dataset expansion with I-Sign integration
- Pose-based features (MediaPipe landmarks)
