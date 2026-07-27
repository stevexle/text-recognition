"""
Composite Hybrid ViT + Transformer Decoder OCR Model.
Wraps HybridViTEncoder and TransformerDecoder into a unified OCR pipeline
supporting parallel training with Teacher Forcing and autoregressive text generation for inference.
"""

import torch
import torch.nn as nn

from src.models.vit_encoder import HybridViTEncoder
from src.models.transformer_decoder import TransformerDecoder


class ViTTransformerOCR(nn.Module):
    """
    Composite Sequence-to-Sequence OCR Model combining:
        - HybridViTEncoder: ConvStem + ViT Encoder Blocks
        - TransformerDecoder: Masked Self-Attention + Cross-Attention Decoder Blocks
    """

    def __init__(
        self,
        vocab_size: int = 200,
        in_channels: int = 3,
        stem_channels: list[int] = None,
        embed_dim: int = 384,
        encoder_depth: int = 6,
        encoder_heads: int = 6,
        decoder_layers: int = 4,
        decoder_heads: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        image_size: tuple[int, int] = (32, 256),
        max_seq_len: int = 256,
        pad_idx: int = 0
    ):
        super().__init__()
        # If custom stem_channels is provided, sync embed_dim with the last channel stage
        if stem_channels is not None:
            embed_dim = stem_channels[-1]
        else:
            stem_channels = [64, 128, embed_dim]
            
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        
        # 1. Feature Extractor & Visual Memory Encoder
        self.encoder = HybridViTEncoder(
            in_channels=in_channels,
            stem_channels=stem_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            image_size=image_size
        )
        
        # 2. Sequence Decoder & Vocabulary Projection Head
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=embed_dim,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx
        )

    def forward(
        self,
        images: torch.Tensor,
        targets_input: torch.Tensor,
        targets_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward Pass for Training (Parallel Teacher Forcing).
        
        Args:
            images: Input image tensor of shape [B, in_channels, H, W]
            targets_input: Shifted-right target token IDs of shape [B, L]
            targets_key_padding_mask: Optional BoolTensor indicating <pad> token positions [B, L]
            
        Returns:
            Logits tensor over vocabulary of shape [B, L, vocab_size]
        """
        # 1. Encode visual features -> Memory Z_enc [B, N, embed_dim]
        memory = self.encoder(images)
        
        # 2. Decode text sequence logits -> [B, L, vocab_size]
        logits = self.decoder(
            tgt_tokens=targets_input,
            memory=memory,
            tgt_key_padding_mask=targets_key_padding_mask
        )
        
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 64,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> torch.Tensor:
        """
        Autoregressive greedy search text generation for Inference.
        
        Args:
            images: Input image tensor of shape [B, in_channels, H, W]
            max_len: Maximum sequence length to generate
            sos_idx: Start-of-Sequence token ID
            eos_idx: End-of-Sequence token ID
            
        Returns:
            Generated token IDs tensor of shape [B, generated_length]
        """
        self.eval()
        B = images.size(0)
        device = images.device
        
        # 1. Encode visual features ONCE -> Memory Z_enc [B, N, embed_dim]
        memory = self.encoder(images)
        
        # 2. Initialize target tokens with <sos> -> Shape [B, 1]
        ys = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        
        # 3. Autoregressive decoding loop
        for _ in range(max_len - 1):
            logits = self.decoder(tgt_tokens=ys, memory=memory)  # Shape: [B, len(ys), vocab_size]
            next_word_logits = logits[:, -1, :]                   # Last token logits: [B, vocab_size]
            next_word = next_word_logits.argmax(dim=-1, keepdim=True)  # Greedy choice: [B, 1]
            
            ys = torch.cat([ys, next_word], dim=1)                # Append predicted token
            
            # Early stopping check: Stop if all samples in batch generated <eos>
            if (ys == eos_idx).any(dim=1).all():
                break
                
        return ys


if __name__ == "__main__":
    # Sanity Check for ViTTransformerOCR
    vocab_size = 187
    batch_size = 2
    seq_len = 20
    
    model = ViTTransformerOCR(
        vocab_size=vocab_size,
        in_channels=3,
        stem_channels=[64, 128, 384],
        embed_dim=384,
        encoder_depth=6,
        encoder_heads=6,
        decoder_layers=4,
        decoder_heads=6,
        image_size=(32, 256)
    )
    
    dummy_images = torch.randn(batch_size, 3, 32, 256)
    dummy_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("--- ViTTransformerOCR Sanity Check ---")
    print(f"Input images shape:        {dummy_images.shape}")
    print(f"Input targets shape:       {dummy_targets.shape}")
    
    # Test Training Forward Pass
    logits = model(dummy_images, dummy_targets)
    print(f"Training logits shape:     {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), "Training forward logits shape mismatch!"
    
    # Test Inference Autoregressive Generation
    generated_tokens = model.generate(dummy_images, max_len=30, sos_idx=1, eos_idx=2)
    print(f"Generated tokens shape:   {generated_tokens.shape}")
    print(f"Total model parameters:    {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Custom Dynamic Hyperparameters (e.g., custom stem_channels=[32, 64, 256])
    custom_model = ViTTransformerOCR(
        vocab_size=vocab_size,
        in_channels=1,
        stem_channels=[32, 64, 256],
        encoder_depth=4,
        encoder_heads=4,
        decoder_layers=3,
        decoder_heads=4,
        image_size=(32, 256)
    )
    dummy_custom_img = torch.randn(2, 1, 32, 256)
    dummy_custom_tgt = torch.randint(0, vocab_size, (2, 15))
    custom_logits = custom_model(dummy_custom_img, dummy_custom_tgt)
    print(f"\nCustom model logits shape: {custom_logits.shape}")
    print(f"Custom model parameters:   {sum(p.numel() for p in custom_model.parameters()):,}")
    assert custom_logits.shape == (2, 15, vocab_size), "Custom model logits shape mismatch!"
    
    print("\nAll ViTTransformerOCR dynamic verification tests passed successfully!")
