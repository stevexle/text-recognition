"""
Model Builder and Registry for Extensible OCR Architectures.
Enables dynamic instantiation of different OCR model families (ViT-Transformer, CRNN, etc.)
based on configuration or checkpoint metadata.
"""

from typing import Callable, Any
import torch.nn as nn

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a new model architecture class/factory into the registry."""
    def decorator(cls_or_fn):
        MODEL_REGISTRY[name.lower()] = cls_or_fn
        return cls_or_fn
    return decorator


def build_model(
    model_cfg: dict[str, Any],
    vocab_size: int,
    pad_idx: int = 0,
    image_size: tuple[int, int] = (32, 256)
) -> nn.Module:
    """
    Build and return an OCR model instance dynamically from configuration.
    
    Args:
        model_cfg: Model configuration dictionary containing 'name' and sub-configs.
        vocab_size: Total vocabulary size.
        pad_idx: Padding token index.
        image_size: Target image dimensions (height, width).
        
    Returns:
        Instantiated nn.Module OCR model.
    """
    model_name = model_cfg.get("name", "vit_transformer").lower()
    
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model architecture '{model_name}'. "
            f"Available registered models: {available}"
        )
        
    builder_fn = MODEL_REGISTRY[model_name]
    return builder_fn(model_cfg=model_cfg, vocab_size=vocab_size, pad_idx=pad_idx, image_size=image_size)


# --- Model Builders ---

@register_model("vit_transformer")
def build_vit_transformer(
    model_cfg: dict[str, Any],
    vocab_size: int,
    pad_idx: int = 0,
    image_size: tuple[int, int] = (32, 256)
) -> nn.Module:
    """Factory builder for Hybrid ViT + Transformer Decoder OCR Model."""
    from src.models.vit_transformer import ViTTransformerOCR
    
    encoder_cfg = model_cfg.get("encoder", {})
    decoder_cfg = model_cfg.get("decoder", {})

    return ViTTransformerOCR(
        vocab_size=vocab_size,
        in_channels=encoder_cfg.get("conv_stem", {}).get("in_channels", 3),
        stem_channels=encoder_cfg.get("conv_stem", {}).get("stem_channels", [64, 128, 384]),
        embed_dim=encoder_cfg.get("embed_dim", 384),
        encoder_depth=encoder_cfg.get("depth", 6),
        encoder_heads=encoder_cfg.get("num_heads", 6),
        decoder_layers=decoder_cfg.get("num_layers", 4),
        decoder_heads=decoder_cfg.get("nhead", 6),
        dim_feedforward=decoder_cfg.get("dim_feedforward", 1536),
        dropout=decoder_cfg.get("dropout", 0.1),
        image_size=image_size,
        max_seq_len=decoder_cfg.get("max_seq_len", 256),
        pad_idx=pad_idx
    )
