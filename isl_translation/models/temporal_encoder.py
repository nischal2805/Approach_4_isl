#!/usr/bin/env python3
"""
ISL Translation - Temporal Encoder

BiLSTM for capturing temporal dependencies in video features.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalEncoder(nn.Module):
    """
    Bidirectional LSTM for temporal encoding of video features.
    
    Input: [B, T, D] - Sequence of frame features
    Output: [B, T, 2*hidden] - Bidirectional hidden states
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.output_dim = hidden_dim * self.num_directions
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # Initialize weights for better stability
        self._init_weights()
        
        logger.info(f"TemporalEncoder: input={input_dim}, hidden={hidden_dim}, "
                   f"layers={num_layers}, bidir={bidirectional}, output={self.output_dim}")
    
    def _init_weights(self):
        """Initialize LSTM weights using orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            x: [B, T, D] input sequence
            lengths: [B] sequence lengths for packing (optional)
            
        Returns:
            outputs: [B, T, 2*hidden] hidden states
            (h_n, c_n): final hidden and cell states
        """
        batch_size, seq_len, _ = x.shape
        
        if lengths is not None:
            # Pack padded sequences for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, (h_n, c_n) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (h_n, c_n) = self.lstm(x)
        
        # Apply layer norm
        outputs = self.layer_norm(outputs)
        
        return outputs, (h_n, c_n)
    
    def get_final_hidden(self, h_n: torch.Tensor) -> torch.Tensor:
        """
        Extract final hidden state for decoder initialization.
        
        Args:
            h_n: [num_layers * num_directions, B, hidden]
            
        Returns:
            [B, hidden * num_directions] concatenated final states
        """
        if self.bidirectional:
            # Concatenate forward and backward final states
            h_forward = h_n[-2]  # [B, hidden]
            h_backward = h_n[-1]  # [B, hidden]
            return torch.cat([h_forward, h_backward], dim=-1)  # [B, 2*hidden]
        else:
            return h_n[-1]  # [B, hidden]


class ProjectionLayer(nn.Module):
    """
    Projects encoder outputs to decoder input dimension with positional encoding.
    """
    
    def __init__(self, input_dim: int, output_dim: int, max_seq_len: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, output_dim) * 0.02)
        
        self.output_dim = output_dim
        
        logger.info(f"ProjectionLayer: {input_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim]
            
        Returns:
            [B, T, output_dim] projected features with positional encoding
        """
        seq_len = x.size(1)
        
        x = self.projection(x)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x


class VideoEncoder(nn.Module):
    """
    Complete video encoder: I3D + BiLSTM + Projection.
    """
    
    def __init__(self, i3d_pretrained: Optional[str] = None, freeze_i3d: bool = True,
                 lstm_hidden: int = 512, lstm_layers: int = 2,
                 lstm_dropout: float = 0.3, projection_dim: int = 512):
        super().__init__()
        
        from .i3d_encoder import I3DEncoder
        
        # I3D visual encoder
        self.i3d = I3DEncoder(pretrained_path=i3d_pretrained, freeze=freeze_i3d)
        
        # Temporal encoder
        self.temporal = TemporalEncoder(
            input_dim=self.i3d.output_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            bidirectional=True
        )
        
        # Projection to decoder dimension
        self.projection = ProjectionLayer(
            input_dim=self.temporal.output_dim,
            output_dim=projection_dim
        )
        
        self.output_dim = projection_dim
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"VideoEncoder: {total_params/1e6:.2f}M params, {trainable_params/1e6:.2f}M trainable")
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, 3, T, H, W] RGB video
            
        Returns:
            [B, T', projection_dim] encoded video features
        """
        # Extract frame features
        features = self.i3d(video)  # [B, T', 1024]
        
        # Temporal encoding
        encoded, _ = self.temporal(features)  # [B, T', 1024]
        
        # Project to decoder dimension
        projected = self.projection(encoded)  # [B, T', projection_dim]
        
        return projected
