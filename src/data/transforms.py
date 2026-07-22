"""
Image transformations and preprocessing pipeline for OCR images.
"""

import torch
import torchvision.transforms as T
from PIL import Image


def get_ocr_transforms(image_size: tuple[int, int] = (32, 256), is_train: bool = True):
    """
    Build PyTorch image transformation pipeline for OCR images.
    
    Args:
        image_size: Target tuple (Height, Width) e.g. (32, 256).
        is_train: Flag to apply training augmentations.
        
    Returns:
        torchvision.transforms.Compose pipeline.
    """
    height, width = image_size
    
    transforms_list = [
        T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to range [-1, 1]
    ]
    
    return T.Compose(transforms_list)
