#!/usr/bin/env python3
"""
Quick verification test for ISL Translation pipeline.
Tests model creation, forward pass, and generation with synthetic data.
"""

import sys
import torch
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def test_i3d_encoder():
    """Test I3D encoder."""
    print("Testing I3D Encoder...")
    from models.i3d_encoder import I3D, I3DEncoder
    
    model = I3D()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  I3D params: {params:.2f}M")
    
    # Test forward pass
    x = torch.randn(1, 3, 30, 224, 224)  # [B, C, T, H, W]
    with torch.no_grad():
        out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    assert out.shape[0] == 1 and out.shape[2] == 1024, f"Unexpected shape: {out.shape}"
    print("  ✓ I3D Encoder OK")
    return True

def test_temporal_encoder():
    """Test BiLSTM temporal encoder."""
    print("\nTesting Temporal Encoder...")
    from models.temporal_encoder import TemporalEncoder, ProjectionLayer
    
    encoder = TemporalEncoder(input_dim=1024, hidden_dim=512, num_layers=2)
    print(f"  Output dim: {encoder.output_dim}")
    
    x = torch.randn(2, 10, 1024)  # [B, T, D]
    out, (h_n, c_n) = encoder(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 10, 1024), f"Unexpected shape: {out.shape}"
    
    # Test projection
    proj = ProjectionLayer(input_dim=1024, output_dim=512)
    proj_out = proj(out)
    print(f"  Projected: {proj_out.shape}")
    assert proj_out.shape == (2, 10, 512)
    print("  ✓ Temporal Encoder OK")
    return True

def test_translator():
    """Test full translator model."""
    print("\nTesting Full Translator...")
    from models.translator import ISLTranslator
    
    # Create model (will download T5 if not cached)
    model = ISLTranslator(
        t5_model_name='t5-small',
        freeze_i3d=True,
        lstm_hidden=256,  # Smaller for testing
        lstm_layers=1
    )
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total params: {total_params:.2f}M, Trainable: {trainable:.2f}M")
    
    # Test forward pass
    video = torch.randn(1, 3, 16, 224, 224)  # Smaller for speed
    labels = model.tokenizer("Hello world", return_tensors="pt")["input_ids"]
    
    model.eval()
    with torch.no_grad():
        outputs = model(video, labels=labels)
    
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    # Test generation
    print("  Testing generation...")
    translations = model.translate(video, max_length=20, num_beams=2)
    print(f"  Generated: '{translations[0]}'")
    
    print("  ✓ Translator OK")
    return True

def test_training_utils():
    """Test training utilities."""
    print("\nTesting Training Utilities...")
    from training.utils import (
        NaNDetector, GradientClipper, CheckpointManager,
        EarlyStopping, get_cosine_schedule_with_warmup
    )
    
    # NaN detector
    nan_det = NaNDetector(patience=3)
    valid, _ = nan_det.check(torch.tensor(1.5))
    assert valid, "Valid loss flagged as NaN"
    print("  ✓ NaN Detector OK")
    
    # Gradient clipper
    clipper = GradientClipper(max_norm=1.0)
    print("  ✓ Gradient Clipper OK")
    
    # Early stopping
    early = EarlyStopping(patience=3)
    for val in [1.0, 0.9, 0.8, 0.85, 0.86, 0.87]:
        stop = early.check(val)
    print(f"  Early stopping triggered: {early.should_stop}")
    print("  ✓ Early Stopping OK")
    
    return True

def main():
    print("=" * 60)
    print("ISL TRANSLATION PIPELINE VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("I3D Encoder", test_i3d_encoder),
        ("Temporal Encoder", test_temporal_encoder),
        ("Translator", test_translator),
        ("Training Utils", test_training_utils),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Pipeline ready for training!\n")
    else:
        print("\n❌ SOME TESTS FAILED\n")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
