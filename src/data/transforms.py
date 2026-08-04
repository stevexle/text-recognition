"""
Image transformations and preprocessing pipeline for OCR images.
"""

import torch
import torchvision.transforms as T
from PIL import Image


def get_ocr_transforms(image_size: tuple[int, int] = (32, 256), is_train: bool = True):
    """
    Build PyTorch image transformation pipeline for OCR images.
    Includes stroke-level & visual augmentations during training.
    
    Args:
        image_size: Target tuple (Height, Width) e.g. (32, 256).
        is_train: Flag to apply training augmentations.
        
    Returns:
        torchvision.transforms.Compose pipeline.
    """
    height, width = image_size
    
    if is_train:
        transforms_list = [
            T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomApply([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
            ], p=0.4),
            T.RandomApply([
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
            ], p=0.3),
            T.RandomAffine(degrees=(-2, 2), scale=(0.95, 1.05), shear=(-2, 2), fill=255),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    else:
        transforms_list = [
            T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    
    return T.Compose(transforms_list)
