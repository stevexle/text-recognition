"""
High-Performance ONNX Runtime OCR Predictor for ViT-Transformer Text Recognition.

Executes Encoder & Decoder entirely via ONNX Runtime with hardware acceleration
(CUDAExecutionProvider / CoreMLExecutionProvider / CPUExecutionProvider).
Supports single image, vectorized batch inference, and asynchronous ASGI prediction.
Features:
  - Auto-detection and pre-registration of NVIDIA CUDA & cuDNN shared libraries on Linux.
  - Fused in-place zero-allocation image preprocessing directly into batch buffers.
  - Preallocated token buffers eliminating array reallocations during decoding.
  - Encoder run once per batch; vectorized autoregressive greedy decoder loop.
"""

import asyncio
import os
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Union

# Auto-detect and register NVIDIA CUDA, cuDNN, and TensorRT shared libraries on Linux
# MUST be executed BEFORE importing onnxruntime or torch
if sys.platform == "linux" and "ORT_CUDA_LOADED" not in os.environ:
    import site
    extra_dirs = []
    try:
        for sp in site.getsitepackages():
            nv = os.path.join(sp, "nvidia")
            if os.path.isdir(nv):
                for sub in ["cuda_runtime", "cublas", "cudnn", "cufft", "curand", "tensorrt"]:
                    d = os.path.join(nv, sub, "lib")
                    if os.path.isdir(d):
                        extra_dirs.append(d)
        if extra_dirs:
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            new_ld = ":".join(extra_dirs + ([current_ld] if current_ld else []))
            os.environ["LD_LIBRARY_PATH"] = new_ld
            os.environ["ORT_CUDA_LOADED"] = "1"
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception:
                pass
    except Exception:
        pass

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

        # 5. Pre-computed ImageNet Normalization Constants
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.scale = 1.0 / (255.0 * std)
        self.bias = -mean / std

    def _resolve_providers(self, user_providers: Optional[List[str]]) -> List[str]:
        if user_providers is not None:
            return user_providers

        available = ort.get_available_providers()
        selected = []

        if "CUDAExecutionProvider" in available:
            # Check CUDA provider options
            cuda_options = {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2 GB
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
            selected.append(("CUDAExecutionProvider", cuda_options))

        selected.append("CPUExecutionProvider")
        return selected

    def _preprocess_image_into(
        self,
        img_input: Union[str, Path, Image.Image, np.ndarray],
        out_buffer: np.ndarray,
    ) -> None:
        """
        Zero-allocation in-place image preprocessing:
        Reads, resizes, normalizes, and writes directly into target slice out_buffer (3, H, W).
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
                img_rgb = img_input
            else:
                img_rgb = img_input[:, :, :3]
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")

        target_h, target_w = self.image_size
        resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # In-place channel assignment and fused normalization: (resized * scale) + bias
        out_buffer[0] = resized[:, :, 0] * self.scale[0] + self.bias[0]
        out_buffer[1] = resized[:, :, 1] * self.scale[1] + self.bias[1]
        out_buffer[2] = resized[:, :, 2] * self.scale[2] + self.bias[2]

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
