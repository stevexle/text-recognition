"""
High-Performance ONNX Runtime OCR Predictor for ViT-Transformer Text Recognition.

Executes Encoder & Decoder entirely via ONNX Runtime with hardware acceleration
(CUDAExecutionProvider / CoreMLExecutionProvider / CPUExecutionProvider).
Supports single image, vectorized batch inference, and asynchronous ASGI prediction.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from src.data.tokenizer import Tokenizer


class OCRPredictorONNX:
    """Production-grade ONNX Runtime Predictor for ViT-Transformer OCR."""

    def __init__(
        self,
        encoder_onnx: str = "weights/onnx/encoder.onnx",
        decoder_onnx: str = "weights/onnx/decoder.onnx",
        vocab_path: str = "checkpoints/vocab.json",
        image_size: tuple[int, int] = (32, 256),
        max_len: int = 64,
        providers: Optional[List[str]] = None,
    ):
        self.image_size = image_size
        self.max_len = max_len
        self.tokenizer = Tokenizer.load(vocab_path)

        # 1. Resolve hardware execution providers
        self.providers = self._resolve_providers(providers)

        # 2. Session Options
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = min(4, max(1, (os.cpu_count() or 4) // 2))

        # 3. Initialize Sessions
        enc_p = Path(encoder_onnx)
        dec_p = Path(decoder_onnx)
        if not enc_p.exists():
            raise FileNotFoundError(f"Encoder ONNX model not found: {enc_p}")
        if not dec_p.exists():
            raise FileNotFoundError(f"Decoder ONNX model not found: {dec_p}")

        self.encoder_sess = ort.InferenceSession(str(enc_p), sess_options=sess_opts, providers=self.providers)
        self.decoder_sess = ort.InferenceSession(str(dec_p), sess_options=sess_opts, providers=self.providers)

        self.enc_input_name = self.encoder_sess.get_inputs()[0].name
        self.dec_tokens_name = self.decoder_sess.get_inputs()[0].name
        self.dec_memory_name = self.decoder_sess.get_inputs()[1].name

        # 4. Precomputed normalization parameters (ImageNet standard)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.scale = (1.0 / (255.0 * std)).astype(np.float32)
        self.bias = (-mean / std).astype(np.float32)

    @staticmethod
    def _resolve_providers(requested: Optional[List[str]]) -> List[str]:
        if requested is not None:
            return requested
        available = ort.get_available_providers()
        resolved = []
        if "CUDAExecutionProvider" in available:
            resolved.append("CUDAExecutionProvider")
        # CoreML has limitations with dynamic batch sizes on Mac, prefer CPU on macOS
        # if "CoreMLExecutionProvider" in available:
        #     resolved.append("CoreMLExecutionProvider")
        resolved.append("CPUExecutionProvider")
        return resolved

    def _preprocess_image(self, img_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Resize image to (H=32, W=256), convert to RGB, apply fused normalization,
        and output contiguous array of shape (3, H, W).
        """
        if isinstance(img_input, (str, Path)):
            raw_bgr = cv2.imread(str(img_input))
            if raw_bgr is None:
                raise ValueError(f"Could not load image from path: {img_input}")
            img_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, Image.Image):
            img_rgb = np.array(img_input.convert("RGB"))
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            elif img_input.shape[2] == 3:
                img_rgb = img_input  # assumes RGB if numpy, or convert if BGR
            else:
                img_rgb = img_input[:, :, :3]
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")

        target_h, target_w = self.image_size
        resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Fused normalization: (img * scale) + bias
        norm = resized.astype(np.float32)
        norm = np.multiply(norm, self.scale, out=norm)
        norm = np.add(norm, self.bias, out=norm)

        # (H, W, C) -> (C, H, W)
        return np.ascontiguousarray(np.transpose(norm, (2, 0, 1)))

    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> str:
        """Run OCR text recognition on a single cropped text image."""
        results = self.predict_batch([image], batch_size=1)
        return results[0] if results else ""

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 16,
    ) -> List[str]:
        """
        Run OCR text recognition across a batch of cropped text images.
        Uses vectorized batch greedy search to decode all samples simultaneously.
        """
        if not images:
            return []

        all_preds = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            tensors = [self._preprocess_image(im) for im in chunk]
            batch_tensor = np.stack(tensors, axis=0)  # Shape: (B, 3, 32, 256)
            b_size = batch_tensor.shape[0]

            # 1. Run ViT Encoder ONCE -> Memory (B, 256, 384)
            memory = self.encoder_sess.run(None, {self.enc_input_name: batch_tensor})[0]

            # 2. Vectorized Greedy Decoding Loop
            tokens = np.full((b_size, 1), self.tokenizer.sos_id, dtype=np.int64)
            finished = np.zeros(b_size, dtype=bool)

            for _ in range(self.max_len):
                logits = self.decoder_sess.run(
                    None,
                    {self.dec_tokens_name: tokens, self.dec_memory_name: memory},
                )[0]

                # Argmax on last position for each batch element
                next_tokens = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)  # (B, 1)
                tokens = np.concatenate([tokens, next_tokens], axis=1)

                # Check for EOS
                is_eos = (next_tokens.squeeze(-1) == self.tokenizer.eos_id)
                finished = finished | is_eos
                if finished.all():
                    break

            # 3. Decode token sequences into strings
            for b in range(b_size):
                tok_seq = tokens[b].tolist()
                text = self.tokenizer.decode(tok_seq)
                all_preds.append(text)

        return all_preds

    async def predict_async(self, image: Union[str, Path, Image.Image, np.ndarray]) -> str:
        """Asynchronous non-blocking prediction for FastAPI/ASGI servers."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, image)

    async def predict_batch_async(
        self,
        images: Sequence[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 16,
    ) -> List[str]:
        """Asynchronous batch prediction for FastAPI/ASGI servers."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict_batch, images, batch_size)
