"""
PyTorch Dataset wrapper for OCR text recognition.
Supports loading from both local CSV files / DataFrames and HuggingFace Dataset splits directly.
Includes automatic fallback for corrupted or missing samples in large-scale datasets.
"""

import os
import io
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.tokenizer import Tokenizer
from src.data.transforms import get_ocr_transforms


class OCRDataset(Dataset):
    """
    Modular OCR Dataset wrapper supporting local CSV metadata, DataFrames,
    and direct HuggingFace Dataset splits (memory-mapped Arrow format).
    Includes automatic fallback to next sample if corrupt image is encountered.
    """

    def __init__(
        self,
        data_source,
        img_dir: str = None,
        tokenizer: Tokenizer = None,
        transform=None,
        image_size: tuple[int, int] = (32, 256),
        is_train: bool = True
    ):
        """
        Args:
            data_source: Path to CSV file (str), pandas DataFrame, or HuggingFace Dataset split.
            img_dir: Base directory containing local image files (used when data_source is CSV/DataFrame).
            tokenizer: Initialized Tokenizer instance.
            transform: Optional torchvision transform pipeline.
            image_size: Target tuple (Height, Width) for image resizing.
            is_train: Training mode flag.
        """
        self.tokenizer = tokenizer
        self.transform = transform or get_ocr_transforms(image_size=image_size, is_train=is_train)
        self.img_dir = img_dir

        if isinstance(data_source, str):
            text_col = "text"
            self.df = pd.read_csv(data_source).dropna(subset=[text_col]).reset_index(drop=True)
            self.hf_dataset = None
        elif isinstance(data_source, pd.DataFrame):
            text_col = "text"
            self.df = data_source.dropna(subset=[text_col]).reset_index(drop=True)
            self.hf_dataset = None
        else:
            # Direct HuggingFace Dataset split
            self.hf_dataset = data_source
            self.df = None

    def __len__(self) -> int:
        if self.hf_dataset is not None:
            return len(self.hf_dataset)
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        try:
            if self.hf_dataset is not None:
                item = self.hf_dataset[idx]
                
                # Robust text key extraction across HF formats ('txt', 'text', 'label', 'transcription')
                raw_text = ""
                for key in ["txt", "text", "label", "transcription", "ground_truth"]:
                    if key in item and item[key] is not None:
                        raw_text = str(item[key]).strip()
                        break

                # Robust image key extraction ('jpg', 'image', 'png', 'img')
                raw_image = None
                for key in ["jpg", "image", "png", "img", "jpeg"]:
                    if key in item and item[key] is not None:
                        raw_image = item[key]
                        break

                # Option A Fallback: If image is missing, skip to next index
                if raw_image is None or raw_text == "":
                    return self.__getitem__((idx + 1) % len(self))

                if isinstance(raw_image, bytes):
                    image = Image.open(io.BytesIO(raw_image))
                elif isinstance(raw_image, dict) and "bytes" in raw_image:
                    image = Image.open(io.BytesIO(raw_image["bytes"]))
                else:
                    image = raw_image
            else:
                row = self.df.iloc[idx]
                raw_text = str(row["text"]).strip()
                
                # Retrieve relative image path column ('image' or 'image_path')
                rel_img_path = str(row.get("image", row.get("image_path", "")))
                
                if self.img_dir:
                    full_img_path = os.path.join(self.img_dir, rel_img_path)
                else:
                    full_img_path = rel_img_path

                if not os.path.exists(full_img_path):
                    full_img_path = os.path.abspath(rel_img_path)

                image = Image.open(full_img_path)

            # Ensure RGB format
            if hasattr(image, "mode") and image.mode != "RGB":
                image = image.convert("RGB")

            # Apply torchvision transforms -> Tensor [3, H, W]
            image_tensor = self.transform(image)

            # Encode text string to token IDs
            label_ids = self.tokenizer.encode(raw_text, add_special_tokens=True)

            return {
                "image": image_tensor,
                "label_ids": torch.tensor(label_ids, dtype=torch.long),
                "text": raw_text
            }
        except Exception:
            # Option A Fallback: On any load/convert error, gracefully skip to next sample index
            return self.__getitem__((idx + 1) % len(self))
