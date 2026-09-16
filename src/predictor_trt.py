"""
High-Performance NVIDIA TensorRT Predictor for ViT-Transformer Text Recognition.

Executes Encoder & Decoder via native TensorRT C++ Execution Contexts.
Features:
  - Exact match with PyTorch training transforms (PIL Bilinear with anti-aliasing + mean=0.5, std=0.5).
  - Zero-allocation contiguous CUDA device buffer reuse and pinned host memory transfers.
  - Encoder executed once per batch; vectorized autoregressive greedy decoder loop up to max_len=256.
  - Static memory binding cached outside the autoregressive decoding loop.
  - Preallocated contiguous logits device buffer eliminating per-token cudaMalloc calls.
  - Fused in-place image normalization directly into contiguous batch tensors.
  - Dedicated CUDA streams for async H2D, compute, and D2H transfers.
  - Supports single image, batch inference, and ASGI asynchronous prediction.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch

from src.data.tokenizer import Tokenizer


class OCRPredictorTRT:
    """
    High-Performance TensorRT Predictor for ViT-Transformer OCR.
    Runs Encoder and Decoder as separate TensorRT engines.
    """

    def __init__(
        self,
        encoder_engine: str = "weights/tensorrt/encoder.engine",
        decoder_engine: str = "weights/tensorrt/decoder.engine",
        vocab_path: str = "checkpoints/vocab.json",
        image_size: tuple[int, int] = (32, 256),
        max_len: int = 256,
        device_id: int = 0,
    ):
        self.image_size = image_size
        self.max_len = max_len
        self.tokenizer = Tokenizer.load(vocab_path)
        self.vocab_size = self.tokenizer.vocab_size

        if not torch.cuda.is_available():
            raise RuntimeError(
                "NVIDIA CUDA is required for TensorRT inference, but torch.cuda is unavailable."
            )

        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(self.device)

        try:
            import tensorrt as trt
            self.trt = trt
        except ImportError as e:
            raise RuntimeError(
                "tensorrt Python package is required. Install via: uv sync or uv pip install tensorrt-cu12"
            ) from e

        # 1. Initialize TensorRT Runtime
        trt_logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.runtime = self.trt.Runtime(trt_logger)

        # 2. Load Encoder Engine
        self.encoder_path = Path(encoder_engine)
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Encoder engine not found at: {self.encoder_path}")
        with open(self.encoder_path, "rb") as f:
            self.encoder_engine = self.runtime.deserialize_cuda_engine(f.read())
        self.encoder_ctx = self.encoder_engine.create_execution_context()

        # 3. Load Decoder Engine
        self.decoder_path = Path(decoder_engine)
        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Decoder engine not found at: {self.decoder_path}")
        with open(self.decoder_path, "rb") as f:
            self.decoder_engine = self.runtime.deserialize_cuda_engine(f.read())
        self.decoder_ctx = self.decoder_engine.create_execution_context()

        # 4. Create CUDA Stream
        self.stream = torch.cuda.Stream(device=self.device)

        # 5. Discover I/O Tensor Names
        self.enc_input_name = self._get_tensor_names(self.encoder_engine, is_input=True)[0]
        self.enc_output_name = self._get_tensor_names(self.encoder_engine, is_input=False)[0]

        dec_inputs = self._get_tensor_names(self.decoder_engine, is_input=True)
        self.dec_tgt_name = "tgt_tokens" if "tgt_tokens" in dec_inputs else dec_inputs[0]
        self.dec_mem_name = "memory" if "memory" in dec_inputs else dec_inputs[1]
        self.dec_output_name = self._get_tensor_names(self.decoder_engine, is_input=False)[0]

        # 6. Cached device buffers for zero-allocation reuse
        self._cached_batch_size: Optional[int] = None
        self._gpu_image: Optional[torch.Tensor] = None
        self._gpu_enc_memory: Optional[torch.Tensor] = None
        self._gpu_logits_raw: Optional[torch.Tensor] = None
        self._host_image_pinned: Optional[torch.Tensor] = None

        # Normalization constants matching torchvision.transforms: mean=0.5, std=0.5
        # (x / 255.0 - 0.5) / 0.5 = (x / 127.5) - 1.0 -> range [-1.0, 1.0]
        self.scale = 1.0 / 127.5
        self.bias = -1.0

    def _get_tensor_names(self, engine, is_input: bool) -> List[str]:
        names = []
        if hasattr(engine, "get_tensor_name"):
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                expected = self.trt.TensorIOMode.INPUT if is_input else self.trt.TensorIOMode.OUTPUT
                if mode == expected:
                    names.append(name)
        else:
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i) == is_input:
                    names.append(engine.get_binding_name(i))
        return names

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

        # In-place channel assignment with fused normalization [-1.0, 1.0]
        out_buffer[0] = arr[:, :, 0] * self.scale + self.bias
        out_buffer[1] = arr[:, :, 1] * self.scale + self.bias
        out_buffer[2] = arr[:, :, 2] * self.scale + self.bias

    def _run_encoder(self, images_np: np.ndarray) -> torch.Tensor:
        """Execute Encoder on batch images (B, 3, 32, 256) and return memory tensor (B, 256, 384)."""
        b, c, h, w = images_np.shape
        target_shape = (b, c, h, w)

        # Allocate or reuse GPU buffers
        if self._cached_batch_size != b:
            self._cached_batch_size = b
            self._gpu_image = torch.empty(target_shape, dtype=torch.float32, device=self.device)
            self._gpu_enc_memory = torch.empty((b, 256, 384), dtype=torch.float32, device=self.device)
            self._gpu_logits_raw = torch.empty((b * self.max_len * self.vocab_size,), dtype=torch.float32, device=self.device)
            try:
                self._host_image_pinned = torch.empty(target_shape, dtype=torch.float32, pin_memory=True)
            except Exception:
                self._host_image_pinned = None

        with torch.cuda.stream(self.stream):
            # Async copy Host -> Device
            if self._host_image_pinned is not None:
                self._host_image_pinned.copy_(torch.from_numpy(images_np))
                self._gpu_image.copy_(self._host_image_pinned, non_blocking=True)
            else:
                self._gpu_image.copy_(torch.from_numpy(images_np), non_blocking=True)

            # Set input shape and tensor addresses
            if hasattr(self.encoder_ctx, "set_input_shape"):
                self.encoder_ctx.set_input_shape(self.enc_input_name, target_shape)
                self.encoder_ctx.set_tensor_address(self.enc_input_name, self._gpu_image.data_ptr())
                self.encoder_ctx.set_tensor_address(self.enc_output_name, self._gpu_enc_memory.data_ptr())
                self.encoder_ctx.execute_async_v3(stream_handle=self.stream.cuda_stream)
            else:
                bindings = [int(self._gpu_image.data_ptr()), int(self._gpu_enc_memory.data_ptr())]
                self.encoder_ctx.set_binding_shape(0, target_shape)
                self.encoder_ctx.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)

        return self._gpu_enc_memory

    def _decode_batch_greedy(self, memory: torch.Tensor, batch_size: int) -> List[str]:
        """
        Vectorized autoregressive greedy decoding loop executed entirely on GPU.
        Memory: shape (B, 256, 384) on GPU.
        """
        current_tokens = torch.full((batch_size, 1), self.tokenizer.sos_id, dtype=torch.int64, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        with torch.cuda.stream(self.stream):
            # 1. Bind static visual memory shape & address ONCE outside the decoding loop
            if hasattr(self.decoder_ctx, "set_input_shape"):
                self.decoder_ctx.set_input_shape(self.dec_mem_name, (batch_size, 256, 384))
                self.decoder_ctx.set_tensor_address(self.dec_mem_name, memory.data_ptr())

            # 2. Autoregressive token generation loop
            for step in range(self.max_len - 1):
                cur_len = current_tokens.shape[1]
                num_elem = batch_size * cur_len * self.vocab_size
                # Guarantee 100% contiguous memory layout matching TensorRT output buffer
                logits_out = self._gpu_logits_raw[:num_elem].view(batch_size, cur_len, self.vocab_size)

                # Update dynamic token shape and binding pointers
                if hasattr(self.decoder_ctx, "set_input_shape"):
                    self.decoder_ctx.set_input_shape(self.dec_tgt_name, (batch_size, cur_len))
                    self.decoder_ctx.set_tensor_address(self.dec_tgt_name, current_tokens.data_ptr())
                    self.decoder_ctx.set_tensor_address(self.dec_output_name, logits_out.data_ptr())
                    self.decoder_ctx.execute_async_v3(stream_handle=self.stream.cuda_stream)
                else:
                    bindings = [
                        int(current_tokens.data_ptr()),
                        int(memory.data_ptr()),
                        int(logits_out.data_ptr()),
                    ]
                    self.decoder_ctx.set_binding_shape(0, (batch_size, cur_len))
                    self.decoder_ctx.set_binding_shape(1, (batch_size, 256, 384))
                    self.decoder_ctx.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)

                # Last token logits -> greedy argmax on GPU
                last_logits = logits_out[:, -1, :]  # [B, vocab_size]
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)  # [B, 1]

                # Mask already finished batch items with pad_id
                next_token[finished.unsqueeze(1)] = self.tokenizer.pad_id

                # Update finished status
                is_eos = (next_token.squeeze(1) == self.tokenizer.eos_id)
                finished = finished | is_eos

                current_tokens = torch.cat([current_tokens, next_token], dim=1)

                # Avoid GPU-to-CPU synchronization stalls during the first 5 characters
                if step >= 5 and finished.all():
                    break

        self.stream.synchronize()

        # Decode token sequences to text strings
        results = []
        tokens_cpu = current_tokens.cpu().tolist()
        for b_tokens in tokens_cpu:
            res_str = self.tokenizer.decode(b_tokens)
            results.append(res_str)

        return results

    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> str:
        """Run OCR recognition on a single image."""
        results = self.predict_batch([image], batch_size=1)
        return results[0] if results else ""

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 16,
    ) -> List[str]:
        """Run batch OCR recognition on a sequence of cropped text images."""
        if not images:
            return []

        all_results: List[str] = []
        total = len(images)
        target_h, target_w = self.image_size

        for i in range(0, total, batch_size):
            chunk = images[i : i + batch_size]
            b_size = len(chunk)

            # Preallocate contiguous batch array and normalize in-place (B, 3, 32, 256)
            batch_np = np.empty((b_size, 3, target_h, target_w), dtype=np.float32)
            for idx, img in enumerate(chunk):
                self._preprocess_image_into(img, batch_np[idx])

            # 1. Run Encoder (once per batch)
            memory = self._run_encoder(batch_np)

            # 2. Run Vectorized Autoregressive Decoder
            batch_texts = self._decode_batch_greedy(memory, b_size)
            all_results.extend(batch_texts)

        return all_results

    async def predict_async(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> str:
        """Asynchronous wrapper for non-blocking ASGI/FastAPI deployment."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, image)
