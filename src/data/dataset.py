"""
PyTorch Dataset wrapper for OCR text recognition dataset.
Supports CSV metadata with image paths and ground truth text annotations.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.tokenizer import Tokenizer
from src.data.transforms import get_ocr_transforms


class OCRDataset(Dataset):
    """
    OCR Dataset wrapper loading images and text annotations from a CSV file.
    """

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        tokenizer: Tokenizer,
        transform=None,
        image_size: tuple[int, int] = (32, 256),
        is_train: bool = True
    ):
        """
        Args:
            csv_file: Path to the annotation CSV file.
            img_dir: Base directory containing OCR images.
            tokenizer: Initialized Tokenizer instance.
            transform: Optional torchvision transform pipeline.
            image_size: Target tuple (Height, Width) for image resizing.
            is_train: Training mode flag.
        """
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform or get_ocr_transforms(image_size=image_size, is_train=is_train)
        
        # Load CSV metadata
        self.df = pd.read_csv(csv_file)
        self.df.dropna(subset=["text"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        rel_img_path = str(row["image_path"])
        raw_text = str(row["text"])

        # Resolve image path
        full_img_path = os.path.join(self.img_dir, rel_img_path)
        if not os.path.exists(full_img_path):
            # Fallback check relative to CSV directory
            full_img_path = os.path.join(os.path.dirname(self.csv_file), rel_img_path)

        # Load image
        try:
            image = Image.open(full_img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image at path: {full_img_path}. Error: {e}")

        # Apply image transformations -> Tensor [3, Height, Width]
        image_tensor = self.transform(image)

        # Encode text label -> Token IDs
        label_ids = self.tokenizer.encode(raw_text, add_special_tokens=True)

        return {
            "image": image_tensor,
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "text": raw_text
        }
