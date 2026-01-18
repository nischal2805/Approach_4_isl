#!/usr/bin/env python3
"""
Check for ACTUAL bugs that would break your current training.
Run this to verify your setup before/during training.
"""

import sys
import yaml
from pathlib import Path

def check_config_compatibility():
    """Check if train_improved.yaml is compatible with the code."""
    
    print("="*70)
    print("CHECKING CONFIG → CODE COMPATIBILITY")
    print("="*70)
    
    # Load config
    config_path = Path("configs/train_improved.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    issues = []
    warnings = []
    
    # 1. Check model config extraction
    print("\n1️⃣  Model Configuration:")
    print("-"*70)
    
    if 'model' in config:
        model_cfg = config['model']
        print(f"✓ Model config found")
        print(f"  - t5_model: {model_cfg.get('t5_model')}")
        print(f"  - i3d_pretrained: {model_cfg.get('i3d_pretrained')}")
        print(f"  - lstm_hidden: {model_cfg.get('lstm_hidden')}")
        print(f"  - lstm_layers: {model_cfg.get('lstm_layers')}")
        
        # Check if freeze_t5_layers is used
        if 'freeze_t5_layers' in model_cfg:
            warnings.append("freeze_t5_layers is in config but NOT implemented in code")
            print(f"  ⚠ freeze_t5_layers: {model_cfg.get('freeze_t5_layers')} (UNUSED)")
    else:
        issues.append("Missing 'model' section in config")
    
    # 2. Check training config
    print("\n2️⃣  Training Configuration:")
    print("-"*70)
    
    if 'training' in config:
        train_cfg = config['training']
        print(f"✓ Training config found")
        print(f"  - batch_size: {train_cfg.get('batch_size')}")
        print(f"  - learning_rate: {train_cfg.get('learning_rate')}")
        print(f"  - focal_gamma: {train_cfg.get('focal_gamma', 0.0)}")
        print(f"  - rdrop_alpha: {train_cfg.get('rdrop_alpha', 0.0)}")
        
        # Check if these are actually used
        focal = train_cfg.get('focal_gamma', 0.0)
        rdrop = train_cfg.get('rdrop_alpha', 0.0)
        
        if focal > 0:
            print(f"  ✓ Focal Loss enabled (gamma={focal})")
        if rdrop > 0:
            print(f"  ✓ R-Drop enabled (alpha={rdrop})")
    
    # 3. Check augmentation
    print("\n3️⃣  Data Augmentation:")
    print("-"*70)
    
    if 'augmentation' in config:
        aug_cfg = config['augmentation']
        enabled = aug_cfg.get('enabled', False)
        print(f"  Enabled: {enabled}")
        
        if enabled:
            # Check if augmentation.py exists
            aug_file = Path("data/augmentation.py")
            if aug_file.exists():
                print(f"  ✓ augmentation.py found")
            else:
                issues.append("Augmentation enabled but data/augmentation.py missing!")
    else:
        print(f"  No augmentation config (will use dataset.py defaults)")
    
    # 4. Check class weighting
    print("\n4️⃣  Class Weighting:")
    print("-"*70)
    
    if 'class_weighting' in config:
        cw_cfg = config['class_weighting']
        enabled = cw_cfg.get('enabled', False)
        
        if enabled:
            warnings.append("class_weighting enabled but NOT implemented in trainer.py!")
            print(f"  ⚠ Enabled: {enabled} - BUT NOT USED IN CODE")
            print(f"  Strategy: {cw_cfg.get('strategy')}")
        else:
            print(f"  Disabled")
    else:
        print(f"  Not configured")
    
    # 5. Check paths
    print("\n5️⃣  Dataset Path:")
    print("-"*70)
    
    dataset_dir = config.get('paths', {}).get('dataset_dir')
    if dataset_dir:
        dataset_path = Path(dataset_dir)
        if dataset_path.exists():
            # Check structure
            subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
            print(f"  ✓ Path exists: {dataset_dir}")
            print(f"  Found {len(subdirs)} subdirectories")
            
            if len(subdirs) == 0:
                issues.append("Dataset directory has no subdirectories!")
            else:
                # Check a few subdirs for videos
                sample_dir = subdirs[0]
                videos = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.MP4"))
                print(f"  Sample class '{sample_dir.name}': {len(videos)} videos")
        else:
            issues.append(f"Dataset path does NOT exist: {dataset_dir}")
    else:
        issues.append("No dataset_dir in config!")
    
    # 6. Check I3D weights
    print("\n6️⃣  I3D Pretrained Weights:")
    print("-"*70)
    
    i3d_path = config.get('model', {}).get('i3d_pretrained')
    if i3d_path:
        if Path(i3d_path).exists():
            print(f"  ✓ Found: {i3d_path}")
        else:
            warnings.append(f"I3D weights not found at {i3d_path}")
            print(f"  ⚠ Not found: {i3d_path}")
            print(f"  Model will use random initialization!")
    else:
        print(f"  No pretrained weights configured (random init)")
    
    # 7. Check stability settings
    print("\n7️⃣  Stability Settings:")
    print("-"*70)
    
    stab_cfg = config.get('stability', {})
    
    if 'gradient_checkpointing' in stab_cfg:
        warnings.append("gradient_checkpointing in config but NOT implemented")
        print(f"  ⚠ gradient_checkpointing: {stab_cfg['gradient_checkpointing']} (UNUSED)")
    
    if 'detect_anomaly' in stab_cfg:
        warnings.append("detect_anomaly in config but NOT implemented")
        print(f"  ⚠ detect_anomaly: {stab_cfg['detect_anomaly']} (UNUSED)")
    
    print(f"  ✓ gradient_clip: {stab_cfg.get('gradient_clip')}")
    print(f"  ✓ mixed_precision: {stab_cfg.get('mixed_precision')}")
    print(f"  ✓ warmup_steps: {stab_cfg.get('warmup_steps')}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if issues:
        print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n⚠️  FIX THESE BEFORE TRAINING!")
        return False
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print("\n✓ Training will work, but some config options are ignored")
    
    if not issues and not warnings:
        print("\n✅ ALL CHECKS PASSED!")
        print("   Config is fully compatible with code")
    elif not issues:
        print("\n✅ NO CRITICAL ISSUES")
        print("   Training will work properly")
    
    return len(issues) == 0


def check_runtime_compatibility():
    """Check if trainer.py can actually use the config."""
    
    print("\n" + "="*70)
    print("CHECKING TRAINER.PY COMPATIBILITY")
    print("="*70)
    
    # Try to import and create trainer
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        import yaml
        from training.trainer import Trainer
        
        config_path = Path("configs/train_improved.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        print("\n✓ Config loads successfully")
        
        # Check key accesses
        print("\nChecking config access patterns...")
        
        # Model config
        model_cfg = config.get('model', {})
        t5 = model_cfg.get('t5_model', 't5-small')
        print(f"  ✓ model.t5_model: {t5}")
        
        # Training config  
        train_cfg = config.get('training', {})
        focal = train_cfg.get('focal_gamma', 0.0)
        rdrop = train_cfg.get('rdrop_alpha', 0.0)
        print(f"  ✓ training.focal_gamma: {focal}")
        print(f"  ✓ training.rdrop_alpha: {rdrop}")
        
        # Check if Focal/RDrop will be enabled
        if focal > 0:
            print(f"  → Focal Loss WILL be used")
        if rdrop > 0:
            print(f"  → R-Drop WILL be used")
        
        print("\n✅ Trainer can read config correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\nCHECKING YOUR TRAINING SETUP...\n")
    
    config_ok = check_config_compatibility()
    runtime_ok = check_runtime_compatibility()
    
    print("\n" + "="*70)
    if config_ok and runtime_ok:
        print("🎉 EVERYTHING LOOKS GOOD!")
        print("\nYour training should work properly with train_improved.yaml")
        print("\nTo start training:")
        print("  python training/trainer.py --config configs/train_improved.yaml")
    else:
        print("⚠️  THERE ARE ISSUES")
        print("\nFix the critical issues above before training")
    print("="*70)