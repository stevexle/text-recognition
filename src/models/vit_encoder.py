"""
Hybrid Vision Transformer (Hybrid ViT) Encoder for Image Feature Extraction.
Combines 3-stage Convolutional Stem (Conv2D + BatchNorm + ReLU) with ViT Transformer Encoder Blocks.
"""

import math
import torch
import torch.nn as nn


class ConvStem(nn.Module):
    """
    3-Stage Convolutional Stem to extract fine-grained character strokes and accents
    while smoothly downsampling spatial dimensions before ViT.
    
    Input shape:  [B, in_channels, H, W]     (e.g., [B, 3, 32, 256])
    Output shape: [B, stem_channels[-1], H/8, W/4] (e.g., [B, 384, 4, 64])
    """

    def __init__(self, in_channels: int = 3, stem_channels: list[int] = None):
        super().__init__()
        if stem_channels is None:
            stem_channels = [64, 128, 384]
            
        c1, c2, c3 = stem_channels
        
        # Stage 1: [B, in_channels, H, W] -> [B, c1, H/2, W/2]  (e.g., [B, 3, 32, 256] -> [B, 64, 16, 128])
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: [B, c1, H/2, W/2] -> [B, c2, H/4, W/4]        (e.g., [B, 64, 16, 128] -> [B, 128, 8, 64])
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: [B, c2, H/4, W/4] -> [B, c3, H/8, W/4]        (e.g., [B, 128, 8, 64] -> [B, 384, 4, 64])
        # Note: Stride (2, 1) shrinks height by 2 while preserving horizontal resolution for text lines
        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape:  [B, in_channels, H, W]
        x = self.stage1(x)  # Shape: [B, c1, H/2, W/2]
        x = self.stage2(x)  # Shape: [B, c2, H/4, W/4]
        x = self.stage3(x)  # Shape: [B, c3, H/8, W/4]
        return x


class HybridViTEncoder(nn.Module):
    """
    Hybrid Vision Transformer Encoder.
    
    Architecture:
        1. ConvStem: Feature extraction + spatial downsampling -> [B, embed_dim, H/8, W/4]
        2. Patch Flatten & Permute -> [B, N=(H/8)*(W/4), embed_dim]
        3. 2D Learnable Positional Embedding addition
        4. Transformer Encoder Blocks (Multi-Head Self-Attention)
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: list[int] = None,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        image_size: tuple[int, int] = (32, 256)
    ):
        super().__init__()
        if stem_channels is None:
            stem_channels = [64, 128, embed_dim]
        else:
            embed_dim = stem_channels[-1]
            
        self.embed_dim = embed_dim
        
        # 1. Fully Dynamic Convolutional Feature Extractor
        self.conv_stem = ConvStem(in_channels=in_channels, stem_channels=stem_channels)
        
        # Calculate spatial feature dimensions and patch sequence length N
        feat_h = image_size[0] // 8   # e.g., 32 // 8 = 4
        feat_w = image_size[1] // 4   # e.g., 256 // 4 = 64
        self.num_patches = feat_h * feat_w  # e.g., 4 * 64 = 256
        
        # 2. 2D Learnable Positional Embedding: Shape [1, N, embed_dim]
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 3. ViT Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass.
        
        Args:
            x: Input image tensor of shape [B, in_channels, H, W]
            
        Returns:
            Visual Memory Z_enc tensor of shape [B, N, embed_dim]
        """
        # 1. Conv Stem Feature Extraction -> [B, c3, H/8, W/4]
        feats = self.conv_stem(x)
        
        # 2. Reshape to Patch Sequence -> [B, H/8, W/4, c3] -> [B, N, embed_dim]
        B, C, H, W = feats.shape
        x_patches = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # 3. Add Positional Embeddings (Explicitly expand batch dimension from 1 to B)
        x_patches = x_patches + self.pos_embed.expand(B, -1, -1)
        x_patches = self.pos_drop(x_patches)
        
        # 4. Pass through ViT Encoder Blocks -> [B, N, embed_dim]
        z_enc = self.blocks(x_patches)
        z_enc = self.norm(z_enc)
        
        return z_enc


if __name__ == "__main__":
    # Quick sanity check with default hyperparameters
    model = HybridViTEncoder(
        in_channels=3,
        stem_channels=[64, 128, 384],
        embed_dim=384,
        depth=6,
        num_heads=6,
        image_size=(32, 256)
    )
    dummy_input = torch.randn(2, 3, 32, 256)
    output = model(dummy_input)
    
    print("--- HybridViTEncoder Sanity Check ---")
    print(f"Input tensor shape:  {dummy_input.shape}")
    print(f"Output Z_enc shape: {output.shape}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert output.shape == (2, 256, 384), "Output shape mismatch!"
    
    # Quick sanity check with custom dynamic hyperparameters
    custom_model = HybridViTEncoder(
        in_channels=1,
        stem_channels=[32, 64, 256],
        embed_dim=256,
        depth=4,
        num_heads=4,
        image_size=(64, 512)
    )
    dummy_input_custom = torch.randn(2, 1, 64, 512)
    output_custom = custom_model(dummy_input_custom)
    print(f"\nCustom Input tensor shape:  {dummy_input_custom.shape}")
    print(f"Custom Output Z_enc shape: {output_custom.shape}")
    print(f"Custom model parameters:   {sum(p.numel() for p in custom_model.parameters()):,}")
    assert output_custom.shape == (2, (64//8)*(512//4), 256), "Custom output shape mismatch!"
    
    print("\nAll HybridViTEncoder dynamic verification tests passed!")
