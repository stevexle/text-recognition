"""
Models module for OCR Architectures.
"""

from src.models.builder import build_model, register_model, MODEL_REGISTRY
from src.models.vit_transformer import ViTTransformerOCR

__all__ = ["build_model", "register_model", "MODEL_REGISTRY", "ViTTransformerOCR"]
