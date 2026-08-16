"""
Standalone OCR Predictor Module.
Provides OCRPredictor class for easy loading of checkpoints and running inference
on single images, batches, or raw PIL Image objects.
"""

from pathlib import Path
from typing import Union
from PIL import Image
import torch

from src.utils import load_checkpoint
from src.data.tokenizer import Tokenizer
from src.data.transforms import get_ocr_transforms
from src.models.builder import build_model


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class OCRPredictor:
    """High-level Predictor for ViT-Transformer OCR."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        vocab_path: str = "checkpoints/vocab.json",
        device: Union[str, torch.device] = None
    ):
        if device is None:
            self.device = get_default_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.tokenizer = Tokenizer.load(vocab_path)
        
        ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
        config = ckpt.get("config", {})
        
        self.image_size = tuple(config.get("dataset", {}).get("image_size", [32, 256]))
        self.max_label_length = config.get("dataset", {}).get("max_label_length", 256)
        
        # Instantiate model dynamically via Model Registry/Factory
        model_cfg = config.get("model", {"name": "vit_transformer"})
        self.model = build_model(
            model_cfg=model_cfg,
            vocab_size=self.tokenizer.vocab_size,
            pad_idx=self.tokenizer.pad_id,
            image_size=self.image_size
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_ocr_transforms(image_size=self.image_size, is_train=False)

    @torch.no_grad()
    def predict(self, image_input: Union[str, Path, Image.Image], beam_size: int = 1) -> str:
        """
        Run OCR inference on a single image.
        
        Args:
            image_input: Path string, Path object, or PIL Image.
            beam_size: Beam search width (1 = greedy decoding).
            
        Returns:
            Predicted text string.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        gen_tokens = self.model.generate(
            tensor,
            max_len=self.max_label_length,
            sos_idx=self.tokenizer.sos_id,
            eos_idx=self.tokenizer.eos_id,
            beam_size=beam_size
        )
        return self.tokenizer.decode(gen_tokens[0].cpu().tolist())

    @torch.no_grad()
    def predict_batch(
        self,
        image_inputs: list[Union[str, Path, Image.Image]],
        beam_size: int = 1,
        batch_size: int = 16
    ) -> list[str]:
        """
        Run OCR inference on a list of images in batches.
        """
        results = []
        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i:i + batch_size]
            tensors = []
            for item in batch:
                if isinstance(item, (str, Path)):
                    img = Image.open(item).convert("RGB")
                else:
                    img = item.convert("RGB")
                tensors.append(self.transform(img))

            batch_tensor = torch.stack(tensors, dim=0).to(self.device)
            gen_tokens = self.model.generate(
                batch_tensor,
                max_len=self.max_label_length,
                sos_idx=self.tokenizer.sos_id,
                eos_idx=self.tokenizer.eos_id,
                beam_size=beam_size
            )
            for tokens in gen_tokens:
                results.append(self.tokenizer.decode(tokens.cpu().tolist()))
        return results
