#!/usr/bin/env python3
"""
ISL Translation - Training Loop

Complete training pipeline with all stability measures.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import T5Tokenizer

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ISLTranslator, create_model
from data.dataset import ISLCSLTRDataset, create_dataloaders
from training.utils import (
    NaNDetector, GradientClipper, CheckpointManager, MetricsLogger,
    EarlyStopping, get_cosine_schedule_with_warmup, seed_everything, count_parameters
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def compute_bleu(predictions: list, references: list) -> float:
    """Compute BLEU score."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        score = bleu.corpus_score(predictions, [references])
        return score.score
    except ImportError:
        # Fallback to simple BLEU
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs = [[ref.split()] for ref in references]
        hyps = [pred.split() for pred in predictions]
        smoothie = SmoothingFunction().method4
        return corpus_bleu(refs, hyps, smoothing_function=smoothie) * 100


class Trainer:
    """Complete training loop with all stability measures."""
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Reproducibility
        seed_everything(42)
        
        # Create model
        logger.info("Creating model...")
        self.model = create_model(config['model'], device)
        self.tokenizer = self.model.tokenizer
        
        # Count parameters
        total, trainable = count_parameters(self.model)
        logger.info(f"Parameters: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config, self.tokenizer
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * config['training']['epochs']
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['stability']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config['stability']['mixed_precision'] else None
        self.use_amp = config['stability']['mixed_precision']
        
        # Stability helpers
        self.nan_detector = NaNDetector(patience=config['stability']['nan_patience'])
        self.gradient_clipper = GradientClipper(max_norm=config['stability']['gradient_clip'])
        
        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            config['paths']['checkpoint_dir'],
            max_checkpoints=5
        )
        
        # Metrics
        self.metrics_logger = MetricsLogger(config['paths']['log_dir'])
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            mode='min'
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.accumulation_steps = config['training']['accumulation_steps']
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            video = batch['video'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(video, labels=labels, attention_mask=attention_mask)
                    loss = outputs['loss'] / self.accumulation_steps
            else:
                outputs = self.model(video, labels=labels, attention_mask=attention_mask)
                loss = outputs['loss'] / self.accumulation_steps
            
            # Check for NaN
            is_valid, safe_loss = self.nan_detector.check(loss)
            if not is_valid:
                pbar.set_postfix({'loss': 'NaN', 'skipped': True})
                continue
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(safe_loss).backward()
            else:
                safe_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Unscale before clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                grad_norm = self.gradient_clipper.clip(self.model)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate loss
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'nan_stats': self.nan_detector.get_stats(),
            'clip_stats': self.gradient_clipper.get_stats()
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_references = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            video = batch['video'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(video, labels=labels, attention_mask=attention_mask)
            else:
                outputs = self.model(video, labels=labels, attention_mask=attention_mask)
            
            if not torch.isnan(outputs['loss']):
                total_loss += outputs['loss'].item()
                num_batches += 1
            
            # Generate predictions for BLEU
            if num_batches <= 10:  # Only generate for first 10 batches (expensive)
                predictions = self.model.translate(video, num_beams=2, max_length=50)
                all_predictions.extend(predictions)
                all_references.extend(batch['text'])
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute BLEU
        bleu = 0.0
        if all_predictions:
            try:
                bleu = compute_bleu(all_predictions, all_references)
            except Exception as e:
                logger.warning(f"BLEU computation failed: {e}")
        
        return {
            'loss': avg_loss,
            'bleu': bleu
        }
    
    def train(self):
        """Full training loop."""
        logger.info("Starting training...")
        logger.info(f"Config: {self.config['training']}")
        
        num_epochs = self.config['training']['epochs']
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log
            self.metrics_logger.log_epoch(self.current_epoch, train_metrics, val_metrics)
            
            logger.info(
                f"Epoch {self.current_epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val BLEU: {val_metrics['bleu']:.2f}"
            )
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Save checkpoint
            if epoch % self.config['training']['save_every_n_epochs'] == 0 or is_best:
                self.checkpoint_manager.save(
                    self.model, self.optimizer, self.scheduler, self.scaler,
                    self.current_epoch, val_metrics, is_best
                )
            
            # Early stopping
            if self.early_stopping.check(val_metrics['loss']):
                logger.info("Early stopping triggered")
                break
        
        # Save final metrics
        self.metrics_logger.save()
        logger.info("Training complete!")
        
        return self.best_val_loss
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set."""
        logger.info("Evaluating on test set...")
        
        # Load best model
        self.checkpoint_manager.load(self.model)
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_references = []
        
        for batch in tqdm(self.test_loader, desc="Testing"):
            video = batch['video'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = self.model(video, labels=labels, attention_mask=attention_mask)
            
            if not torch.isnan(outputs['loss']):
                total_loss += outputs['loss'].item()
                num_batches += 1
            
            # Generate
            predictions = self.model.translate(video, num_beams=4, max_length=50)
            all_predictions.extend(predictions)
            all_references.extend(batch['text'])
        
        avg_loss = total_loss / max(num_batches, 1)
        bleu = compute_bleu(all_predictions, all_references) if all_predictions else 0.0
        
        logger.info(f"Test Loss: {avg_loss:.4f}, Test BLEU: {bleu:.2f}")
        
        # Print some examples
        logger.info("\nSample predictions:")
        for i in range(min(5, len(all_predictions))):
            logger.info(f"  Ref: {all_references[i]}")
            logger.info(f"  Pred: {all_predictions[i]}")
            logger.info("")
        
        return {
            'loss': avg_loss,
            'bleu': bleu,
            'predictions': all_predictions[:10],
            'references': all_references[:10]
        }


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train ISL Translation Model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create trainer
    trainer = Trainer(config, device=args.device)
    
    # Resume if specified
    if args.resume:
        trainer.checkpoint_manager.load(
            trainer.model, trainer.optimizer, 
            trainer.scheduler, trainer.scaler,
            args.resume
        )
    
    if args.evaluate_only:
        trainer.evaluate()
    else:
        trainer.train()
        trainer.evaluate()


if __name__ == '__main__':
    main()
