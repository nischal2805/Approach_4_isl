#!/usr/bin/env python3
"""
ISL Translation - Dataset and DataLoader

Handles loading ISL-CSLTR dataset with video preprocessing.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from transformers import T5Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ISLCSLTRDataset(Dataset):
    """
    Dataset for ISL-CSLTR (Indian Sign Language - Continuous Sign Language 
    Translation and Recognition).
    
    Expected structure:
        dataset_dir/
        ├── videos/
        │   ├── video_001.mp4
        │   ├── video_002.mp4
        │   └── ...
        └── annotations.json  (or .csv)
            [{"video": "video_001.mp4", "text": "Hello my name is John"}, ...]
    """
    
    def __init__(self, 
                 dataset_dir: str,
                 split: str = 'train',
                 num_frames: int = 30,
                 frame_size: int = 224,
                 tokenizer: Optional[T5Tokenizer] = None,
                 max_target_length: int = 50,
                 transform: Optional[Callable] = None):
        """
        Args:
            dataset_dir: Path to dataset directory
            split: 'train', 'val', or 'test'
            num_frames: Number of frames to sample from each video
            frame_size: Frame height/width after resizing
            tokenizer: T5 tokenizer for text encoding
            max_target_length: Maximum target sequence length
            transform: Optional transform for video frames
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.max_target_length = max_target_length
        self.transform = transform
        
        # Load tokenizer
        self.tokenizer = tokenizer or T5Tokenizer.from_pretrained('t5-small')
        
        # Load annotations
        self.samples = self._load_annotations()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from JSON or CSV file."""
        samples = []
        
        # Try JSON first
        json_path = self.dataset_dir / 'annotations.json'
        csv_path = self.dataset_dir / 'annotations.csv'
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get('samples', [])
        elif csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                samples.append({
                    'video': row.get('video', row.get('video_path', row.get('filename'))),
                    'text': row.get('text', row.get('translation', row.get('label')))
                })
        else:
            # Try to infer from directory structure
            samples = self._infer_from_structure()
        
        # Filter for split
        if 'split' in samples[0] if samples else {}:
            samples = [s for s in samples if s.get('split') == self.split]
        else:
            # Random split if not specified
            samples = self._create_split(samples)
        
        return samples
    
    def _infer_from_structure(self) -> List[Dict]:
        """Infer samples from directory structure."""
        samples = []
        videos_dir = self.dataset_dir / 'videos'
        
        if not videos_dir.exists():
            videos_dir = self.dataset_dir
        
        for video_path in videos_dir.glob('*.mp4'):
            # Try to find corresponding text file
            text_path = video_path.with_suffix('.txt')
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            else:
                text = video_path.stem.replace('_', ' ')  # Fallback to filename
            
            samples.append({
                'video': str(video_path),
                'text': text
            })
        
        return samples
    
    def _create_split(self, samples: List[Dict]) -> List[Dict]:
        """Create train/val/test split."""
        import random
        random.seed(42)
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        if self.split == 'train':
            return shuffled[:train_end]
        elif self.split == 'val':
            return shuffled[train_end:val_end]
        else:  # test
            return shuffled[val_end:]
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess video.
        
        Returns:
            [3, T, H, W] tensor
        """
        # Handle relative paths
        if not os.path.isabs(video_path):
            video_path = self.dataset_dir / video_path
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Failed to open video: {video_path}")
            # Return zeros if video fails to load
            return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.warning(f"Empty video: {video_path}")
            return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
        
        # Sample frame indices uniformly
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # Use last good frame or zeros
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
                continue
            
            # Resize and convert BGR to RGB
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Stack and normalize: [T, H, W, 3] -> [3, T, H, W]
        video = np.stack(frames, axis=0)  # [T, H, W, 3]
        video = video.transpose(3, 0, 1, 2)  # [3, T, H, W]
        video = video.astype(np.float32) / 255.0
        
        # Normalize to ImageNet stats
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
        video = (video - mean) / std
        
        return torch.from_numpy(video).float()
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize target text."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            dict with 'video', 'labels', 'attention_mask', 'text'
        """
        sample = self.samples[idx]
        
        # Load video
        video = self._load_video(sample['video'])
        
        if self.transform:
            video = self.transform(video)
        
        # Tokenize text
        text_encoding = self._tokenize_text(sample['text'])
        
        return {
            'video': video,
            'labels': text_encoding['input_ids'],
            'attention_mask': text_encoding['attention_mask'],
            'text': sample['text']
        }


def create_dataloaders(config: dict, tokenizer: T5Tokenizer = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders from config."""
    dataset_dir = config['paths']['dataset_dir']
    
    train_dataset = ISLCSLTRDataset(
        dataset_dir=dataset_dir,
        split='train',
        num_frames=config['data']['num_frames'],
        frame_size=config['data']['frame_size'],
        tokenizer=tokenizer,
        max_target_length=config['model'].get('max_target_length', 50)
    )
    
    val_dataset = ISLCSLTRDataset(
        dataset_dir=dataset_dir,
        split='val',
        num_frames=config['data']['num_frames'],
        frame_size=config['data']['frame_size'],
        tokenizer=tokenizer,
        max_target_length=config['model'].get('max_target_length', 50)
    )
    
    test_dataset = ISLCSLTRDataset(
        dataset_dir=dataset_dir,
        split='test',
        num_frames=config['data']['num_frames'],
        frame_size=config['data']['frame_size'],
        tokenizer=tokenizer,
        max_target_length=config['model'].get('max_target_length', 50)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
