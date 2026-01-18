#!/usr/bin/env python3
"""
ISL Translation - Mobile Export & Quantization

Exports model to ONNX with INT8 quantization for mobile deployment.
Target: Reduce 660MB model to ~60MB.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


class EncoderWrapper(nn.Module):
    """Wrapper for video encoder (I3D + BiLSTM) for ONNX export."""
    
    def __init__(self, translator):
        super().__init__()
        self.video_encoder = translator.video_encoder
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, 3, T, H, W]
        Returns:
            [B, T', D] encoded features
        """
        return self.video_encoder(video)


def export_encoder_onnx(translator, output_path: str, num_frames: int = 30):
    """Export video encoder to ONNX."""
    logger.info("Exporting encoder to ONNX...")
    
    encoder = EncoderWrapper(translator)
    encoder.eval()
    
    # Dummy input
    dummy_video = torch.randn(1, 3, num_frames, 224, 224)
    
    # Export
    torch.onnx.export(
        encoder,
        dummy_video,
        output_path,
        input_names=['video'],
        output_names=['features'],
        dynamic_axes={
            'video': {0: 'batch_size', 2: 'num_frames'},
            'features': {0: 'batch_size', 1: 'seq_len'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    logger.info(f"Encoder exported to {output_path}")
    return output_path


def quantize_onnx_model(onnx_path: str, output_path: str):
    """Apply INT8 dynamic quantization to ONNX model."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        logger.info("Applying INT8 quantization...")
        
        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QInt8,
            optimize_model=True
        )
        
        # Compare sizes
        orig_size = os.path.getsize(onnx_path) / 1e6
        quant_size = os.path.getsize(output_path) / 1e6
        reduction = (1 - quant_size / orig_size) * 100
        
        logger.info(f"Original: {orig_size:.1f}MB -> Quantized: {quant_size:.1f}MB ({reduction:.1f}% reduction)")
        return output_path
        
    except ImportError:
        logger.error("onnxruntime-extensions not installed. Run: pip install onnxruntime-extensions")
        return None


def export_for_mobile(checkpoint_path: str, output_dir: str, num_frames: int = 30, test: bool = False):
    """
    Full export pipeline for mobile deployment.
    
    Creates:
        - encoder.onnx (FP32)
        - encoder_int8.onnx (INT8 quantized)
        - model_info.json (metadata)
    """
    from models import ISLTranslator
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {checkpoint_path}...")
    
    translator = ISLTranslator(
        t5_model_name='t5-small',
        freeze_i3d=True,
        lstm_hidden=512,
        lstm_layers=2
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        translator.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Checkpoint loaded")
    else:
        logger.warning(f"No checkpoint at {checkpoint_path}, using random weights")
    
    translator.eval()
    
    # Export encoder
    encoder_path = output_dir / "encoder.onnx"
    export_encoder_onnx(translator, str(encoder_path), num_frames)
    
    # Quantize
    encoder_int8_path = output_dir / "encoder_int8.onnx"
    quantize_onnx_model(str(encoder_path), str(encoder_int8_path))
    
    # Save T5 tokenizer for mobile
    tokenizer_path = output_dir / "tokenizer"
    translator.tokenizer.save_pretrained(str(tokenizer_path))
    logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    # Export T5 decoder separately (FP16 for size reduction)
    logger.info("Exporting T5 decoder...")
    t5_path = output_dir / "t5_decoder.pt"
    
    # Save only decoder weights in FP16
    t5_state = {k: v.half() for k, v in translator.t5.state_dict().items()}
    torch.save(t5_state, t5_path)
    
    t5_size = os.path.getsize(t5_path) / 1e6
    logger.info(f"T5 decoder saved: {t5_size:.1f}MB")
    
    # Create metadata
    import json
    metadata = {
        "num_frames": num_frames,
        "frame_size": 224,
        "encoder_file": "encoder_int8.onnx",
        "decoder_file": "t5_decoder.pt",
        "tokenizer_dir": "tokenizer",
        "max_length": 50
    }
    
    with open(output_dir / "model_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    total_size = sum(
        os.path.getsize(f) for f in output_dir.glob("*") 
        if f.is_file() and f.suffix in ['.onnx', '.pt']
    ) / 1e6
    
    logger.info("=" * 50)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total size: {total_size:.1f}MB")
    logger.info("Files:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size = os.path.getsize(f) / 1e6
            logger.info(f"  {f.name}: {size:.1f}MB")
    
    # Test if requested
    if test:
        test_exported_model(output_dir, translator, num_frames)
    
    return output_dir


def test_exported_model(output_dir: Path, original_model, num_frames: int):
    """Compare original and quantized model outputs."""
    try:
        import onnxruntime as ort
        
        logger.info("\nTesting exported model...")
        
        # Load ONNX model
        encoder_path = output_dir / "encoder_int8.onnx"
        session = ort.InferenceSession(str(encoder_path))
        
        # Create test input
        test_video = torch.randn(1, 3, num_frames, 224, 224)
        
        # Original output
        original_model.eval()
        with torch.no_grad():
            orig_features = original_model.video_encoder(test_video)
        
        # ONNX output
        onnx_features = session.run(
            None, 
            {'video': test_video.numpy()}
        )[0]
        
        # Compare
        orig_np = orig_features.numpy()
        diff = np.abs(orig_np - onnx_features).mean()
        max_diff = np.abs(orig_np - onnx_features).max()
        
        logger.info(f"Mean absolute difference: {diff:.6f}")
        logger.info(f"Max absolute difference: {max_diff:.6f}")
        
        if diff < 0.1:
            logger.info("✓ Quantization test PASSED")
        else:
            logger.warning("⚠ Quantization introduced significant differences")
            
    except ImportError:
        logger.warning("onnxruntime not installed, skipping test")


def main():
    parser = argparse.ArgumentParser(description='Export ISL model for mobile')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='mobile_export',
                        help='Output directory')
    parser.add_argument('--frames', type=int, default=30,
                        help='Number of frames')
    parser.add_argument('--test', action='store_true',
                        help='Test exported model')
    args = parser.parse_args()
    
    export_for_mobile(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        num_frames=args.frames,
        test=args.test
    )


if __name__ == '__main__':
    main()
