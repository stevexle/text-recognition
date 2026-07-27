"""
Autoregressive Transformer Decoder for Text Generation.
Implements Token & Positional Embeddings, Causal Masking, and Cross-Attention with Visual Memory.
"""

import math
import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder module for OCR sequence generation.
    
    Architecture:
        1. Token Embedding Layer: Token IDs [B, L] -> Vectors [B, L, d_model]
        2. 1D Learnable Positional Embedding addition
        3. Causal Masking (Look-ahead Masking)
        4. Transformer Decoder Layers (Masked Self-Attention + Multi-Head Cross-Attention with Visual Memory)
        5. Output Head: Linear Projection [B, L, d_model] -> Logits [B, L, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int = 200,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        
        # 1. Token & Positional Embeddings
        # Token Embedding weight matrix shape: [vocab_size, d_model]; Input [B, L] -> Output [B, L, d_model]
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # Positional Embedding tensor shape: [1, max_seq_len, d_model]
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 2. Transformer Decoder Blocks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 3. Output Projection Head -> Vocab Probabilities
        self.head = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate a causal square mask for target sequence to prevent attending to future tokens.
        
        Args:
            sz: Sequence length L
            device: Compute device
            
        Returns:
            Float Causal Mask tensor of shape [L, L] with -inf on upper triangle.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for training with Teacher Forcing.
        
        Args:
            tgt_tokens: Target token IDs tensor of shape [B, L]
            memory: Visual Memory Z_enc tensor from Encoder of shape [B, N, d_model]
            tgt_key_padding_mask: Optional BoolTensor indicating <pad> positions of shape [B, L]
            
        Returns:
            Logits tensor of shape [B, L, vocab_size]
        """
        B, L = tgt_tokens.shape
        device = tgt_tokens.device
        
        # Safety assertion: Ensure input sequence length L does not exceed max_seq_len
        assert L <= self.pos_embedding.size(1), (
            f"Input sequence length L={L} exceeds maximum positional embedding limit max_seq_len={self.pos_embedding.size(1)}!"
        )
        
        # 1. Token & Positional Embeddings -> [B, L, d_model]
        # Input tgt_tokens [B, L] -> Token Embedding [B, L, d_model]
        tgt_emb = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)  # Shape: [B, L, d_model]
        pos_emb = self.pos_embedding[:, :L, :]  # Slice up to current sequence length L, Shape: [1, L, d_model]
        tgt_emb = tgt_emb + pos_emb.expand(B, -1, -1)  # Shape: [B, L, d_model]
        tgt_emb = self.pos_drop(tgt_emb)
        
        # 2. Generate Causal Look-ahead Mask -> [L, L]
        tgt_mask = self.generate_square_subsequent_mask(L, device=device)
        
        # 3. Transformer Decoder Blocks -> [B, L, d_model]
        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        dec_out = self.norm(dec_out)
        
        # 4. Linear Output Projection Head -> [B, L, vocab_size]
        logits = self.head(dec_out)
        
        return logits


if __name__ == "__main__":
    # Quick sanity check
    vocab_size = 187
    d_model = 384
    batch_size = 2
    seq_len = 20
    num_patches = 256
    
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=6,
        num_layers=4
    )
    
    dummy_tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_memory = torch.randn(batch_size, num_patches, d_model)
    
    logits = decoder(dummy_tgt, dummy_memory)
    
    print("--- TransformerDecoder Sanity Check ---")
    print(f"Target tokens input shape: {dummy_tgt.shape}")
    print(f"Visual memory input shape:  {dummy_memory.shape}")
    print(f"Output logits shape:        {logits.shape}")
    print(f"Total decoder parameters:   {sum(p.numel() for p in decoder.parameters()):,}")
    
    assert logits.shape == (batch_size, seq_len, vocab_size), "Decoder output shape mismatch!"
    print("\nAll TransformerDecoder verification tests passed!")
