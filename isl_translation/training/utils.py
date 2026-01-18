#!/usr/bin/env python3
"""
ISL Translation - Training Utilities

Includes:
- Gradient clipping
- NaN protection
- Mixed precision helpers
- Checkpointing
- Metrics logging
"""

import os
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NaNDetector:
    """Detects and handles NaN/Inf values during training."""
    
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.nan_count = 0
        self.total_nan = 0
        
    def check(self, loss: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        """
        Check if loss is valid.
        
        Returns:
            (is_valid, safe_loss)
        """
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_count += 1
            self.total_nan += 1
            logger.warning(f"NaN/Inf detected! Count: {self.nan_count}/{self.patience}")
            
            if self.nan_count >= self.patience:
                raise RuntimeError(f"Too many consecutive NaN losses ({self.patience}). Stopping training.")
            
            return False, torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        self.nan_count = 0  # Reset on valid loss
        return True, loss
    
    def get_stats(self) -> Dict[str, int]:
        return {"total_nan": self.total_nan, "current_streak": self.nan_count}


class GradientClipper:
    """Handles gradient clipping with logging."""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
        self.clip_count = 0
        self.total_calls = 0
        
    def clip(self, model: nn.Module) -> float:
        """Clip gradients and return the original norm."""
        self.total_calls += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        
        if grad_norm > self.max_norm:
            self.clip_count += 1
            
        return grad_norm.item()
    
    def get_stats(self) -> Dict[str, Any]:
        clip_rate = self.clip_count / max(self.total_calls, 1)
        return {"clip_rate": clip_rate, "clip_count": self.clip_count}


class CheckpointManager:
    """Manages model checkpoints with best model tracking."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_metric = float('inf')
        self.checkpoints = []
        
    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             scheduler: Any, scaler: Optional[GradScaler],
             epoch: int, metrics: Dict[str, float], is_best: bool = False) -> str:
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        filename = f"checkpoint_epoch_{epoch:03d}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)
        logger.info(f"Saved checkpoint: {filepath}")
        
        # Clean up old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_ckpt = self.checkpoints.pop(0)
            if old_ckpt.exists() and "best" not in str(old_ckpt):
                old_ckpt.unlink()
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved! Metric: {metrics.get('val_loss', 'N/A')}")
            
        return str(filepath)
    
    def load(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Any = None, scaler: Optional[GradScaler] = None,
             checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint."""
        if checkpoint_path is None:
            # Load best model
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            if not checkpoint_path.exists():
                # Load latest
                checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
                if not checkpoints:
                    raise FileNotFoundError("No checkpoints found")
                checkpoint_path = checkpoints[-1]
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


class MetricsLogger:
    """Logs and tracks training metrics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {'train': [], 'val': []}
        self.current_epoch = {}
        
    def log(self, phase: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for a phase."""
        entry = {'step': step, **metrics}
        if phase not in self.history:
            self.history[phase] = []
        self.history[phase].append(entry)
        
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch summary."""
        self.current_epoch = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        }
        
        # Print summary
        train_loss = train_metrics.get('loss', 0)
        val_loss = val_metrics.get('loss', 0)
        val_bleu = val_metrics.get('bleu', 0)
        
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_bleu={val_bleu:.2f}"
        )
        
    def save(self):
        """Save metrics history to file."""
        import json
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)


class EarlyStopping:
    """Early stopping with patience."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
        
    def check(self, value: float) -> bool:
        """Check if training should stop."""
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
            
        if improved:
            self.best_value = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
            return False


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, 
                                     num_training_steps: int, 
                                     num_cycles: float = 0.5):
    """Create cosine learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """
    Focal Loss for seq2seq - down-weights easy/common examples.
    Helps prevent mode collapse to frequent predictions.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, 
                 ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, T, vocab_size]
            targets: [B, T]
        Returns:
            Focal loss scalar
        """
        vocab_size = logits.size(-1)
        
        # Flatten
        logits_flat = logits.view(-1, vocab_size)  # [B*T, V]
        targets_flat = targets.view(-1)  # [B*T]
        
        # Create mask
        mask = targets_flat != self.ignore_index
        
        # Get probabilities
        log_probs = torch.log_softmax(logits_flat, dim=-1)
        probs = torch.softmax(logits_flat, dim=-1)
        
        # Label smoothing
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(probs)
            smooth_targets.fill_(self.label_smoothing / (vocab_size - 1))
            smooth_targets.scatter_(1, targets_flat.unsqueeze(1).clamp(0), 1 - self.label_smoothing)
            
            # Cross entropy with smoothing
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
            pt = (smooth_targets * probs).sum(dim=-1)
        else:
            # Standard cross entropy
            ce_loss = torch.nn.functional.cross_entropy(
                logits_flat, targets_flat.clamp(0), reduction='none'
            )
            pt = probs.gather(1, targets_flat.unsqueeze(1).clamp(0)).squeeze(1)
        
        # Focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Mask and average
        focal_loss = focal_loss * mask.float()
        return focal_loss.sum() / mask.float().sum().clamp(min=1)


class RDropLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks.
    
    Adds KL divergence between two forward passes with different dropout.
    Reduces overfitting and mode collapse.
    
    Paper: https://arxiv.org/abs/2106.14448
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight of the KL divergence term
        """
        super().__init__()
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor, 
                base_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute R-Drop loss.
        
        Args:
            logits1: First forward pass logits [B, T, V]
            logits2: Second forward pass logits [B, T, V]
            base_loss: Average of the two CE losses
            
        Returns:
            Total loss with R-Drop regularization
        """
        # Flatten for KL computation
        p = torch.log_softmax(logits1.view(-1, logits1.size(-1)), dim=-1)
        q = torch.softmax(logits2.view(-1, logits2.size(-1)), dim=-1)
        
        p_rev = torch.log_softmax(logits2.view(-1, logits2.size(-1)), dim=-1)
        q_rev = torch.softmax(logits1.view(-1, logits1.size(-1)), dim=-1)
        
        # Symmetric KL divergence
        kl_loss = (self.kl_div(p, q) + self.kl_div(p_rev, q_rev)) / 2
        
        return base_loss + self.alpha * kl_loss


def compute_class_weights(dataset, num_classes: int = None) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced datasets.
    
    Args:
        dataset: Dataset with samples containing 'text' field
        num_classes: Number of unique classes (auto-detected if None)
        
    Returns:
        torch.Tensor of class weights
    """
    from collections import Counter
    
    # Count sentence frequencies
    texts = [s['text'] for s in dataset.samples]
    counts = Counter(texts)
    
    # Get unique classes
    unique_texts = list(counts.keys())
    if num_classes is None:
        num_classes = len(unique_texts)
    
    # Compute inverse frequency weights
    total = len(texts)
    weights = []
    for text in unique_texts:
        freq = counts[text] / total
        weight = 1.0 / (freq * num_classes)  # Inverse frequency
        weights.append(weight)
    
    # Normalize so mean weight = 1
    weights = torch.tensor(weights)
    weights = weights / weights.mean()
    
    logger.info(f"Class weights computed: min={weights.min():.2f}, max={weights.max():.2f}")
    return weights


def compute_rouge(predictions: list, references: list) -> Dict[str, float]:
    """
    Compute ROUGE scores for translation evaluation.
    
    Returns:
        Dict with rouge1, rouge2, rougeL scores
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            scores['rouge1'] += result['rouge1'].fmeasure
            scores['rouge2'] += result['rouge2'].fmeasure
            scores['rougeL'] += result['rougeL'].fmeasure
        
        n = len(predictions)
        return {k: v / n * 100 for k, v in scores.items()}
    except ImportError:
        logger.warning("rouge_score not installed. Run: pip install rouge-score")
        return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}


def compute_wer(predictions: list, references: list) -> float:
    """
    Compute Word Error Rate (WER).
    
    WER = (Substitutions + Deletions + Insertions) / Reference Words
    Lower is better, 0 = perfect.
    """
    total_wer = 0
    total_words = 0
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.lower().split()
        ref_words = ref.lower().split()
        
        # Simple Levenshtein distance at word level
        m, n = len(ref_words), len(pred_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == pred_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        total_wer += dp[m][n]
        total_words += len(ref_words)
    
    return (total_wer / max(total_words, 1)) * 100


def compute_exact_match(predictions: list, references: list) -> float:
    """Compute exact match accuracy (case-insensitive)."""
    correct = sum(
        1 for p, r in zip(predictions, references) 
        if p.lower().strip() == r.lower().strip()
    )
    return correct / len(predictions) * 100 if predictions else 0
