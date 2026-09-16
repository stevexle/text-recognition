"""
High-Performance ONNX Runtime OCR Predictor for ViT-Transformer Text Recognition.

Executes Encoder & Decoder via ONNX Runtime with hardware acceleration
(CUDAExecutionProvider / CPUExecutionProvider).
Features:
  - Exact match with PyTorch training transforms (PIL Bilinear with anti-aliasing + mean=0.5, std=0.5).
  - Fused in-place zero-allocation image preprocessing directly into batch buffers.
  - Preallocated token buffers eliminating array reallocations during decoding.
  - Encoder run once per batch; vectorized autoregressive greedy decoder loop up to max_len=256.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

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
        max_len: int = 256,
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

        # 3. Create Inference Sessions
        self.encoder_path = Path(encoder_onnx)
        self.decoder_path = Path(decoder_onnx)
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Encoder ONNX model not found: {self.encoder_path}")
        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Decoder ONNX model not found: {self.decoder_path}")

        self.encoder_sess = ort.InferenceSession(
            str(self.encoder_path), sess_opts, providers=self.providers
        )
        self.decoder_sess = ort.InferenceSession(
            str(self.decoder_path), sess_opts, providers=self.providers
        )

        # 4. Cache Input & Output Node Names
        self.enc_input_name = self.encoder_sess.get_inputs()[0].name
        self.enc_output_name = self.encoder_sess.get_outputs()[0].name

        dec_in = self.decoder_sess.get_inputs()
        self.dec_tokens_name = dec_in[0].name
        self.dec_memory_name = dec_in[1].name
        self.dec_output_name = self.decoder_sess.get_outputs()[0].name

        # 5. Normalization constants matching torchvision.transforms: mean=0.5, std=0.5
        # (x / 255.0 - 0.5) / 0.5 = (x / 127.5) - 1.0 -> range [-1.0, 1.0]
        self.scale = 1.0 / 127.5
        self.bias = -1.0

    def _resolve_providers(self, user_providers: Optional[List[str]]) -> List[str]:
        if user_providers is not None:
            return user_providers

        available = ort.get_available_providers()
        selected = []

        if "CUDAExecutionProvider" in available:
            selected.append("CUDAExecutionProvider")

        selected.append("CPUExecutionProvider")
        return selected

    def _preprocess_image_into(
        self,
        img_input: Union[str, Path, Image.Image, np.ndarray],
        out_buffer: np.ndarray,
    ) -> None:
        """
        Preprocess image to match PyTorch training transform:
        Resize to (target_w, target_h) using PIL Bilinear interpolation (with anti-aliasing),
        and apply normalization with mean=0.5, std=0.5 -> range [-1.0, 1.0].
        """
        if isinstance(img_input, (str, Path)):
            pil_img = Image.open(str(img_input)).convert("RGB")
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                pil_img = Image.fromarray(img_input).convert("RGB")
            elif img_input.shape[2] == 3:
                pil_img = Image.fromarray(img_input)
            else:
                pil_img = Image.fromarray(img_input[:, :, :3])
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")

        target_h, target_w = self.image_size
        resized = pil_img.resize((target_w, target_h), resample=Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32)

        # Fused normalization: (x / 127.5) - 1.0 -> range [-1.0, 1.0]
        out_buffer[0] = arr[:, :, 0] * self.scale + self.bias
        out_buffer[1] = arr[:, :, 1] * self.scale + self.bias
        out_buffer[2] = arr[:, :, 2] * self.scale + self.bias

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
        target_h, target_w = self.image_size

        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            b_size = len(chunk)

            # 1. Preallocate batch input tensor once and fill in-place (B, 3, 32, 256)
            batch_tensor = np.empty((b_size, 3, target_h, target_w), dtype=np.float32)
            for idx, im in enumerate(chunk):
                self._preprocess_image_into(im, batch_tensor[idx])

            # 2. Run ViT Encoder ONCE per batch -> Visual Memory (B, 256, 384)
            memory = self.encoder_sess.run(
                [self.enc_output_name],
                {self.enc_input_name: batch_tensor},
            )[0]

            # 3. Preallocated Vectorized Greedy Decoding Buffer
            tokens = np.empty((b_size, self.max_len), dtype=np.int64)
            tokens[:, 0] = self.tokenizer.sos_id
            finished = np.zeros(b_size, dtype=bool)
            actual_len = 1

            for step in range(self.max_len - 1):
                cur_tokens = np.ascontiguousarray(tokens[:, : step + 1])
                logits = self.decoder_sess.run(
                    [self.dec_output_name],
                    {self.dec_tokens_name: cur_tokens, self.dec_memory_name: memory},
                )[0]

                # Vectorized argmax on last token position: (B,)
                next_tokens = np.argmax(logits[:, -1, :], axis=-1)
                tokens[:, step + 1] = next_tokens
                actual_len = step + 2

                # Check for EOS
                is_eos = (next_tokens == self.tokenizer.eos_id)
                finished = finished | is_eos
                if step >= 5 and finished.all():
                    break

            # 4. Decode token sequences into text strings
            for b in range(b_size):
                tok_seq = tokens[b, :actual_len].tolist()
                text = self.tokenizer.decode(tok_seq)
                all_preds.append(text)

        return all_preds

    async def predict_async(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> str:
        """Asynchronous execution wrapper for FastAPI/ASGI server endpoints."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, image)
