#!/usr/bin/env python3
"""
ISL Translation - Full Translator Model

End-to-end model: Video Encoder + T5 Decoder
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ISLTranslator(nn.Module):
    """
    Complete ISL to English translation model.
    
    Architecture:
        Video -> I3D -> BiLSTM -> Projection -> T5 Decoder -> English
    """
    
    def __init__(self, 
                 t5_model_name: str = "t5-small",
                 i3d_pretrained: Optional[str] = None,
                 freeze_i3d: bool = True,
                 lstm_hidden: int = 512,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.3,
                 max_target_length: int = 50):
        super().__init__()
        
        self.max_target_length = max_target_length
        
        # Load T5 for decoder
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        
        # Get T5 hidden dimension
        t5_hidden_dim = self.t5.config.d_model  # 512 for t5-small, 768 for t5-base
        
        # Video encoder
        from .temporal_encoder import VideoEncoder
        self.video_encoder = VideoEncoder(
            i3d_pretrained=i3d_pretrained,
            freeze_i3d=freeze_i3d,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            projection_dim=t5_hidden_dim
        )
        
        # Disable T5 encoder (we use our video encoder instead)
        # We'll use T5 as decoder-only by passing encoder_outputs
        
        self._log_model_info()
    
    def _log_model_info(self):
        """Log model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"ISLTranslator initialized:")
        logger.info(f"  Total parameters: {total_params / 1e6:.2f}M")
        logger.info(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
        logger.info(f"  T5 model: {self.t5.config.name_or_path}")
        logger.info(f"  Max target length: {self.max_target_length}")
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to features compatible with T5 decoder.
        
        Args:
            video: [B, 3, T, H, W] RGB video
            
        Returns:
            [B, T', d_model] encoded features
        """
        return self.video_encoder(video)
    
    def forward(self, 
                video: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            video: [B, 3, T, H, W] input video
            labels: [B, L] target token IDs
            attention_mask: [B, L] attention mask for labels
            
        Returns:
            dict with 'loss', 'logits'
        """
        # Encode video
        encoder_outputs = self.encode_video(video)  # [B, T', d_model]
        
        # Create encoder attention mask (all 1s since video is padded uniformly)
        encoder_attention_mask = torch.ones(
            encoder_outputs.shape[:2], 
            device=encoder_outputs.device,
            dtype=torch.long
        )
        
        # Create BaseModelOutput for T5
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs_obj = BaseModelOutput(
            last_hidden_state=encoder_outputs
        )
        
        # Forward through T5 decoder
        outputs = self.t5(
            encoder_outputs=encoder_outputs_obj,
            attention_mask=encoder_attention_mask,
            labels=labels,
            decoder_attention_mask=attention_mask
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    @torch.no_grad()
    def generate(self, 
                 video: torch.Tensor,
                 max_length: Optional[int] = None,
                 num_beams: int = 4,
                 length_penalty: float = 0.6,
                 early_stopping: bool = True,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate translation from video.
        
        Args:
            video: [B, 3, T, H, W] input video
            max_length: maximum generation length
            num_beams: beam search width
            length_penalty: length penalty for beam search
            
        Returns:
            (generated_ids, scores) tuple
        """
        self.eval()
        
        max_length = max_length or self.max_target_length
        
        # Encode video
        encoder_outputs = self.encode_video(video)
        encoder_attention_mask = torch.ones(
            encoder_outputs.shape[:2],
            device=encoder_outputs.device,
            dtype=torch.long
        )
        
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs_obj = BaseModelOutput(
            last_hidden_state=encoder_outputs
        )
        
        # Generate
        generated = self.t5.generate(
            encoder_outputs=encoder_outputs_obj,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs
        )
        
        return generated.sequences, generated.sequences_scores if hasattr(generated, 'sequences_scores') else None
    
    def decode_tokens(self, token_ids: torch.Tensor) -> list:
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    def translate(self, video: torch.Tensor, **kwargs) -> list:
        """
        Convenience method: video -> English text.
        
        Args:
            video: [B, 3, T, H, W] input video
            
        Returns:
            List of translated sentences
        """
        generated_ids, _ = self.generate(video, **kwargs)
        return self.decode_tokens(generated_ids)


def create_model(config: dict, device: str = 'cuda') -> ISLTranslator:
    """Create model from config dictionary."""
    model = ISLTranslator(
        t5_model_name=config.get('t5_model', 't5-small'),
        i3d_pretrained=config.get('i3d_pretrained'),
        freeze_i3d=config.get('freeze_i3d', True),
        lstm_hidden=config.get('lstm_hidden', 512),
        lstm_layers=config.get('lstm_layers', 2),
        lstm_dropout=config.get('lstm_dropout', 0.3),
        max_target_length=config.get('max_target_length', 50)
    )
    
    return model.to(device)
