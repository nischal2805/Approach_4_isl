#!/usr/bin/env python3
"""
Video Augmentation for ISL Translation

Implements various augmentation techniques to improve model robustness:
- Spatial: Random crop, horizontal flip, color jitter, rotation
- Temporal: Random temporal shift, speed variation
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional, Tuple


class VideoAugmentation:
    """
    Video augmentation pipeline for sign language videos.
    
    Designed to be applied during training to increase diversity
    without breaking the semantic meaning of signs.
    """
    
    def __init__(
        self,
        spatial_prob: float = 0.5,
        temporal_prob: float = 0.3,
        color_prob: float = 0.4,
        # Spatial augmentations
        random_crop_scale: Tuple[float, float] = (0.85, 1.0),
        horizontal_flip: bool = True,  # ISL is mostly symmetric
        rotation_degrees: float = 10.0,
        # Color augmentations  
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        # Temporal augmentations
        temporal_shift_ratio: float = 0.1,
        speed_range: Tuple[float, float] = (0.9, 1.1),
    ):
        self.spatial_prob = spatial_prob
        self.temporal_prob = temporal_prob
        self.color_prob = color_prob
        
        self.random_crop_scale = random_crop_scale
        self.horizontal_flip = horizontal_flip
        self.rotation_degrees = rotation_degrees
        
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        
        self.temporal_shift_ratio = temporal_shift_ratio
        self.speed_range = speed_range
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to video tensor.
        
        Args:
            video: [C, T, H, W] tensor (channels, time, height, width)
            
        Returns:
            Augmented video tensor
        """
        # Spatial augmentations
        if random.random() < self.spatial_prob:
            video = self._apply_spatial(video)
        
        # Color augmentations
        if random.random() < self.color_prob:
            video = self._apply_color(video)
        
        # Temporal augmentations
        if random.random() < self.temporal_prob:
            video = self._apply_temporal(video)
        
        return video
    
    def _apply_spatial(self, video: torch.Tensor) -> torch.Tensor:
        """Apply spatial augmentations."""
        C, T, H, W = video.shape
        
        # Random crop and resize
        scale = random.uniform(*self.random_crop_scale)
        new_h, new_w = int(H * scale), int(W * scale)
        
        # Random position
        top = random.randint(0, H - new_h) if H > new_h else 0
        left = random.randint(0, W - new_w) if W > new_w else 0
        
        # Apply crop
        video = video[:, :, top:top+new_h, left:left+new_w]
        
        # Resize back to original size
        # Reshape for F.interpolate: [C*T, 1, H, W] -> interpolate -> reshape back
        video = video.permute(1, 0, 2, 3)  # [T, C, H, W]
        video = F.interpolate(video, size=(H, W), mode='bilinear', align_corners=False)
        video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        # Horizontal flip (safe for most signs, avoid for asymmetric ones)
        if self.horizontal_flip and random.random() < 0.5:
            video = torch.flip(video, dims=[-1])
        
        # Small rotation (be careful - too much rotation changes sign meaning)
        if self.rotation_degrees > 0 and random.random() < 0.3:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            video = self._rotate_video(video, angle)
        
        return video
    
    def _rotate_video(self, video: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate video frames by given angle."""
        C, T, H, W = video.shape
        
        # Create rotation matrix
        theta = torch.tensor(angle * np.pi / 180.0)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0]
        ], dtype=video.dtype).unsqueeze(0)
        
        # Apply rotation frame by frame
        video = video.permute(1, 0, 2, 3)  # [T, C, H, W]
        grid = F.affine_grid(rotation_matrix.expand(T, -1, -1), video.size(), align_corners=False)
        video = F.grid_sample(video, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        return video
    
    def _apply_color(self, video: torch.Tensor) -> torch.Tensor:
        """Apply color augmentations."""
        # Brightness
        if random.random() < 0.5:
            factor = random.uniform(*self.brightness_range)
            video = video * factor
        
        # Contrast
        if random.random() < 0.5:
            factor = random.uniform(*self.contrast_range)
            mean = video.mean()
            video = (video - mean) * factor + mean
        
        # Clamp values
        video = torch.clamp(video, 0, 1)
        
        return video
    
    def _apply_temporal(self, video: torch.Tensor) -> torch.Tensor:
        """Apply temporal augmentations."""
        C, T, H, W = video.shape
        
        # Temporal shift (shift frames left or right, fill with edge)
        if random.random() < 0.5:
            max_shift = int(T * self.temporal_shift_ratio)
            if max_shift > 0:
                shift = random.randint(-max_shift, max_shift)
                if shift > 0:
                    # Shift right, fill left with first frame
                    video = torch.cat([
                        video[:, :1].repeat(1, shift, 1, 1),
                        video[:, :-shift]
                    ], dim=1)
                elif shift < 0:
                    # Shift left, fill right with last frame
                    video = torch.cat([
                        video[:, -shift:],
                        video[:, -1:].repeat(1, -shift, 1, 1)
                    ], dim=1)
        
        # Speed variation (resample temporal dimension)
        if random.random() < 0.5:
            speed = random.uniform(*self.speed_range)
            new_t = int(T / speed)
            
            if new_t != T and new_t > 1:
                # Resample to new_t frames, then back to T
                # Reshape to [C*H*W, 1, T] for 1D temporal interpolation
                video = video.reshape(C * H * W, 1, T)
                video = F.interpolate(video, size=new_t, mode='linear', align_corners=False)
                video = F.interpolate(video, size=T, mode='linear', align_corners=False)
                video = video.reshape(C, H, W, T).permute(0, 3, 1, 2)  # [C, T, H, W]
        
        return video


class MixUp:
    """
    MixUp augmentation for video classification.
    
    Mixes two videos and their labels to create new training samples.
    """
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(
        self, 
        video1: torch.Tensor, 
        video2: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Mix two video-label pairs.
        
        Returns:
            mixed_video, mixed_label, lambda_value
        """
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_video = lam * video1 + (1 - lam) * video2
        # For seq2seq, we can't easily mix labels, so return lambda for loss weighting
        return mixed_video, label1 if lam > 0.5 else label2, lam


class GaussianNoise:
    """Add Gaussian noise to video frames."""
    
    def __init__(self, std: float = 0.05):
        self.std = std
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(video) * self.std
        return torch.clamp(video + noise, 0, 1)


class RandomErasing:
    """
    Random erasing augmentation.
    
    Randomly erases a rectangular region in video frames.
    Useful for robustness to occlusion.
    """
    
    def __init__(
        self, 
        p: float = 0.3,
        scale: Tuple[float, float] = (0.02, 0.1),
        ratio: Tuple[float, float] = (0.3, 3.3)
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return video
        
        C, T, H, W = video.shape
        
        area = H * W
        target_area = random.uniform(*self.scale) * area
        aspect_ratio = random.uniform(*self.ratio)
        
        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if h < H and w < W:
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)
            
            # Erase with random values or zeros
            video[:, :, top:top+h, left:left+w] = torch.rand(C, T, h, w)
        
        return video


def get_train_augmentation(config: Optional[dict] = None) -> VideoAugmentation:
    """
    Get training augmentation pipeline based on config.
    
    Args:
        config: Configuration dict with augmentation parameters
        
    Returns:
        VideoAugmentation instance
    """
    if config is None:
        config = {}
    
    return VideoAugmentation(
        spatial_prob=config.get('spatial_prob', 0.5),
        temporal_prob=config.get('temporal_prob', 0.3),
        color_prob=config.get('color_prob', 0.4),
        horizontal_flip=config.get('horizontal_flip', True),
        rotation_degrees=config.get('rotation_degrees', 8.0),
    )


def get_val_augmentation() -> None:
    """No augmentation for validation."""
    return None


# Test the augmentation
if __name__ == "__main__":
    print("Testing video augmentation...")
    
    # Create dummy video [C, T, H, W]
    dummy_video = torch.rand(3, 16, 112, 112)
    
    aug = VideoAugmentation()
    augmented = aug(dummy_video)
    
    print(f"Input shape: {dummy_video.shape}")
    print(f"Output shape: {augmented.shape}")
    print(f"Values in range [0, 1]: {augmented.min():.3f} - {augmented.max():.3f}")
    print("✓ Augmentation test passed!")
