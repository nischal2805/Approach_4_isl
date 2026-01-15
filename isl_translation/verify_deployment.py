#!/usr/bin/env python3
"""
Comprehensive verification before server deployment.
"""

import sys
import os
from pathlib import Path

def check_file_exists(path, description):
    """Check if file exists and report."""
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    return exists

def check_directory_exists(path, description):
    """Check if directory exists and report."""
    exists = Path(path).is_dir()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    return exists

def main():
    print("=" * 70)
    print("ISL TRANSLATION SYSTEM - PRE-DEPLOYMENT VERIFICATION")
    print("=" * 70)
    
    all_good = True
    
    # 1. Core Training Files
    print("\n1. TRAINING PIPELINE")
    print("-" * 70)
    all_good &= check_file_exists("models/i3d_encoder.py", "I3D Encoder")
    all_good &= check_file_exists("models/temporal_encoder.py", "Temporal Encoder")
    all_good &= check_file_exists("models/translator.py", "Translator Model")
    all_good &= check_file_exists("training/trainer.py", "Training Loop")
    all_good &= check_file_exists("training/utils.py", "Training Utils")
    all_good &= check_file_exists("data/dataset.py", "Dataset Loader")
    
    # 2. Configuration Files
    print("\n2. CONFIGURATION FILES")
    print("-" * 70)
    all_good &= check_file_exists("configs/train_config.yaml", "Full Training Config (A100)")
    all_good &= check_file_exists("configs/train_laptop.yaml", "Laptop Config (Testing)")
    
    # 3. Dataset
    print("\n3. DATASET")
    print("-" * 70)
    dataset_path = "data/isl_clstr/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Sentence_Level"
    all_good &= check_directory_exists(dataset_path, "ISL-CSLTR Videos")
    
    if Path(dataset_path).exists():
        subdirs = [d for d in Path(dataset_path).iterdir() if d.is_dir()]
        print(f"  → Found {len(subdirs)} sentence classes")
        
        # Count videos
        total_videos = 0
        for subdir in subdirs:
            videos = list(subdir.glob("*.mp4")) + list(subdir.glob("*.MP4"))
            total_videos += len(videos)
        print(f"  → Total videos: {total_videos}")
    
    # 4. Trained Checkpoint
    print("\n4. TRAINED MODEL")
    print("-" * 70)
    has_checkpoint = check_file_exists("checkpoints/best_model.pt", "Best Model Checkpoint")
    if has_checkpoint:
        import torch
        ckpt = torch.load("checkpoints/best_model.pt", map_location='cpu')
        print(f"  → Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  → Val Loss: {ckpt.get('metrics', {}).get('val_loss', 'N/A')}")
    
    # 5. API Server
    print("\n5. INFERENCE API")
    print("-" * 70)
    all_good &= check_file_exists("inference/api.py", "FastAPI Server")
    all_good &= check_file_exists("inference/__init__.py", "Inference Package")
    
    # 6. Flutter App
    print("\n6. FLUTTER MOBILE APP")
    print("-" * 70)
    all_good &= check_file_exists("../isl_app/pubspec.yaml", "Flutter Config")
    all_good &= check_file_exists("../isl_app/lib/main.dart", "Flutter Main")
    all_good &= check_file_exists("../isl_app/lib/screens/home_screen.dart", "Home Screen")
    all_good &= check_file_exists("../isl_app/lib/services/translation_service.dart", "API Service")
    
    # 7. Documentation
    print("\n7. DOCUMENTATION")
    print("-" * 70)
    all_good &= check_file_exists("README.md", "Translation README")
    all_good &= check_file_exists("requirements.txt", "Python Requirements")
    all_good &= check_file_exists("../isl_app/README.md", "Flutter README")
    
    # 8. Testing Scripts
    print("\n8. VERIFICATION SCRIPTS")
    print("-" * 70)
    all_good &= check_file_exists("verify_pipeline.py", "Pipeline Verification")
    all_good &= check_file_exists("test_inference.py", "Inference Test")
    
    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("✓ ALL CHECKS PASSED - READY FOR DEPLOYMENT")
    else:
        print("✗ SOME CHECKS FAILED - REVIEW ABOVE")
    print("=" * 70)
    
    # Next steps
    print("\nNEXT STEPS:")
    print("1. Commit and push to git")
    print("2. SSH to A100 server")
    print("3. Clone repo on server")
    print("4. Install dependencies: pip install -r requirements.txt")
    print("5. Download I3D weights: rgb_imagenet.pt")
    print("6. Run training: python training/trainer.py --config configs/train_config.yaml")
    print("\nExpected training time on A100: 24-36 hours for 50 epochs")
    
    return 0 if all_good else 1

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    sys.exit(main())
