#!/usr/bin/env python3
"""Quick test to see what the model actually generates."""

import torch
from pathlib import Path
from models import ISLTranslator

# Load model
print("Loading model...")
model = ISLTranslator(
    t5_model_name='t5-small',
    freeze_i3d=True,
    lstm_hidden=256,
    lstm_layers=1
)

# Load checkpoint
checkpoint_path = "checkpoints/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Val loss: {checkpoint['metrics'].get('val_loss', 'N/A')}")

# Create a dummy video (random noise)
print("\nTesting with random video...")
dummy_video = torch.randn(1, 3, 16, 112, 112)  # [B, C, T, H, W]

# Generate with different settings
print("\n=== Test 1: Default beam search ===")
with torch.no_grad():
    translations = model.translate(dummy_video, num_beams=4, max_length=30)
    print(f"Output: '{translations[0]}'")
    print(f"Length: {len(translations[0])}")

print("\n=== Test 2: Greedy decoding ===")
with torch.no_grad():
    translations = model.translate(dummy_video, num_beams=1, max_length=30)
    print(f"Output: '{translations[0]}'")
    
print("\n=== Test 3: Higher temperature ===")
with torch.no_grad():
    translations = model.translate(dummy_video, num_beams=1, max_length=50, do_sample=True, temperature=1.5)
    print(f"Output: '{translations[0]}'")

print("\n=== Test 4: Check raw logits ===")
with torch.no_grad():
    outputs = model(dummy_video)
    logits = outputs['logits']
    print(f"Logits shape: {logits.shape}")
    print(f"Logits stats - min: {logits.min():.2f}, max: {logits.max():.2f}, mean: {logits.mean():.2f}")
    
    # Decode first few tokens
    predicted_ids = logits[0].argmax(dim=-1)[:10]
    decoded = model.tokenizer.decode(predicted_ids, skip_special_tokens=False)
    print(f"First 10 tokens (raw): {decoded}")
