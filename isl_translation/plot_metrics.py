#!/usr/bin/env python3
"""
Plot training metrics from logs.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_metrics(log_dir="logs"):
    """Plot loss and BLEU from metrics.json"""
    
    metrics_path = Path(log_dir) / "metrics.json"
    
    if not metrics_path.exists():
        print(f"No metrics found at {metrics_path}")
        return
    
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    train_data = data.get('train', [])
    val_data = data.get('val', [])
    
    if not train_data:
        print("No training data found")
        return
    
    # Extract epochs
    epochs = list(range(1, len(train_data) + 1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax1 = axes[0]
    train_losses = [entry.get('loss', 0) for entry in train_data]
    val_losses = [entry.get('loss', 0) for entry in val_data]
    
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: BLEU
    ax2 = axes[1]
    val_bleus = [entry.get('bleu', 0) for entry in val_data]
    
    ax2.plot(epochs, val_bleus, 'g-o', label='Val BLEU', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('BLEU Score', fontsize=12)
    ax2.set_title('Validation BLEU Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(val_bleus) * 1.2 if max(val_bleus) > 0 else 1)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(log_dir) / "training_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    # Show summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total Epochs: {len(epochs)}")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    print(f"Best Val Loss: {min(val_losses):.4f} (Epoch {val_losses.index(min(val_losses)) + 1})")
    print(f"Final Val BLEU: {val_bleus[-1]:.2f}")
    print(f"Best Val BLEU: {max(val_bleus):.2f} (Epoch {val_bleus.index(max(val_bleus)) + 1})")
    print("="*50)

if __name__ == '__main__':
    plot_training_metrics()
