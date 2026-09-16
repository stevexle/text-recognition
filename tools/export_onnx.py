"""
Automated ONNX Model Exporter for ViT-Transformer OCR (Text Recognition).

Exports PyTorch ViTTransformerOCR into two self-contained, optimized ONNX models:
  1. encoder.onnx: HybridViTEncoder (ConvStem + ViT Blocks)
     - Input:  image [B, 3, 32, 256]
     - Output: memory [B, N=256, 384]
  2. decoder.onnx: TransformerDecoder with Dynamic Sequence Length
     - Inputs: tgt_tokens [B, L], memory [B, N=256, 384]
     - Output: logits [B, L, vocab_size]

Supports onnxslim optimization and end-to-end numerical verification against PyTorch.
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.predictor import OCRPredictor
from src.data.transforms import get_ocr_transforms
from src.data.tokenizer import Tokenizer
from PIL import Image


class DynamicMultiheadAttention(nn.Module):
    """
    ONNX & TensorRT-Friendly Multi-Head Attention replacement.
    Maintains exact numerical equivalence with PyTorch nn.MultiheadAttention
    while preserving dynamic sequence length tracing without baking constant shapes.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = True
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        if query is key and key is value:
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)
            v = F.linear(value, w_v, b_v)

        # Reshape to [B, num_heads, L, head_dim] dynamically
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
        return self.out_proj(attn_output), None


def patch_decoder_for_onnx(decoder: nn.Module) -> nn.Module:
    """
    Replace internal PyTorch MultiheadAttention layers with DynamicMultiheadAttention
    to guarantee dynamic sequence length support across ONNX Runtime and TensorRT.
    """
    for layer in decoder.decoder.layers:
        clean_sa = DynamicMultiheadAttention(decoder.d_model, layer.self_attn.num_heads)
        clean_sa.load_state_dict(layer.self_attn.state_dict())
        layer.self_attn = clean_sa

        clean_ca = DynamicMultiheadAttention(decoder.d_model, layer.multihead_attn.num_heads)
        clean_ca.load_state_dict(layer.multihead_attn.state_dict())
        layer.multihead_attn = clean_ca
    return decoder


def optimize_with_onnxslim(onnx_path: str) -> None:
    """Optimize ONNX graph using onnxslim."""
    try:
        import onnxslim
        slimmed_model = onnxslim.slim(onnx_path)
        import onnx
        onnx.save(slimmed_model, onnx_path)
        print(f"  [onnxslim] Successfully optimized: {onnx_path}")
    except Exception as e:
        print(f"  [onnxslim] Skipping optimization for {onnx_path} (note: {e})")


def parse_args():
    parser = argparse.ArgumentParser(description="Export ViT-Transformer OCR to ONNX.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to best_model.pt")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.json", help="Path to vocab.json")
    parser.add_argument("--output-dir", type=str, default="weights/onnx", help="Target ONNX export directory")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-slim", action="store_true", help="Skip onnxslim optimization")
    parser.add_argument("--verify", action="store_true", default=True, help="Run numerical verification against PyTorch")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ViT-Transformer OCR -> ONNX Exporter")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Vocab:      {args.vocab}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Opset:      {args.opset}\n")

    # 1. Load PyTorch model
    print("Loading PyTorch OCR predictor...")
    predictor = OCRPredictor(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device="cpu",
    )
    predictor.model.eval()
    image_size = predictor.image_size
    vocab_size = predictor.tokenizer.vocab_size
    embed_dim = predictor.model.embed_dim

    print(f"  - Image input size: {image_size} (H={image_size[0]}, W={image_size[1]})")
    print(f"  - Embedding dim:    {embed_dim}")
    print(f"  - Vocabulary size:  {vocab_size}\n")

    # 2. Export Encoder
    encoder_path = output_dir / "encoder.onnx"
    print(f"[1/2] Exporting Encoder to {encoder_path}...")
    encoder = predictor.model.encoder
    encoder.eval()

    dummy_img = torch.randn(1, 3, image_size[0], image_size[1])
    torch.onnx.export(
        encoder,
        dummy_img,
        str(encoder_path),
        input_names=["image"],
        output_names=["memory"],
        dynamic_axes={
            "image": {0: "batch"},
            "memory": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"  - Encoder exported successfully! ({os.path.getsize(encoder_path) / 1024 / 1024:.1f} MB)")
    if not args.no_slim:
        optimize_with_onnxslim(str(encoder_path))

    # 3. Export Decoder
    decoder_path = output_dir / "decoder.onnx"
    print(f"\n[2/2] Exporting Decoder to {decoder_path}...")
    decoder = patch_decoder_for_onnx(predictor.model.decoder)
    decoder.eval()

    dummy_tokens = torch.randint(0, vocab_size, (1, 8), dtype=torch.long)
    dummy_memory = torch.randn(1, encoder.num_patches, embed_dim)

    torch.onnx.export(
        decoder,
        (dummy_tokens, dummy_memory),
        str(decoder_path),
        input_names=["tgt_tokens", "memory"],
        output_names=["logits"],
        dynamic_axes={
            "tgt_tokens": {0: "batch", 1: "seq_len"},
            "memory": {0: "batch"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"  - Decoder exported successfully! ({os.path.getsize(decoder_path) / 1024 / 1024:.1f} MB)")
    if not args.no_slim:
        optimize_with_onnxslim(str(decoder_path))

    # 4. Numerical Verification
    if args.verify:
        print("\n" + "=" * 70)
        print("  Running End-to-End Verification (PyTorch vs ONNX Runtime)")
        print("=" * 70)
        import onnxruntime as ort

        enc_sess = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
        dec_sess = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])

        # Numerical diff test
        test_img = torch.randn(2, 3, image_size[0], image_size[1])
        test_tokens = torch.randint(0, vocab_size, (2, 14), dtype=torch.long)

        with torch.no_grad():
            pt_memory = encoder(test_img)
            pt_logits = decoder(test_tokens, pt_memory)

        ort_memory = enc_sess.run(None, {"image": test_img.numpy()})[0]
        ort_logits = dec_sess.run(None, {"tgt_tokens": test_tokens.numpy(), "memory": ort_memory})[0]

        enc_diff = np.max(np.abs(pt_memory.numpy() - ort_memory))
        dec_diff = np.max(np.abs(pt_logits.numpy() - ort_logits))
        print(f"  - Encoder Max Absolute Error: {enc_diff:.6e}")
        print(f"  - Decoder Max Absolute Error: {dec_diff:.6e}")
        assert enc_diff < 1e-4, f"Encoder error too high: {enc_diff}"
        assert dec_diff < 1e-4, f"Decoder error too high: {dec_diff}"

        # Real test image decoding
        test_dir = Path("test/cropped")
        if test_dir.exists():
            test_files = sorted(list(test_dir.glob("*.jpg")))[:4]
            if test_files:
                print("\n  Text Recognition Sample Comparison:")
                transform = get_ocr_transforms(image_size=image_size, is_train=False)
                tokenizer = predictor.tokenizer

                for img_p in test_files:
                    pil_img = Image.open(img_p).convert("RGB")
                    tensor = transform(pil_img).unsqueeze(0)

                    # PyTorch prediction
                    pt_text = predictor.predict(img_p)

                    # ONNX Runtime prediction
                    memory = enc_sess.run(None, {"image": tensor.numpy()})[0]
                    tokens = np.array([[tokenizer.sos_id]], dtype=np.int64)
                    for _ in range(64):
                        logits = dec_sess.run(None, {"tgt_tokens": tokens, "memory": memory})[0]
                        next_tok = int(np.argmax(logits[0, -1, :]))
                        tokens = np.concatenate([tokens, [[next_tok]]], axis=1)
                        if next_tok == tokenizer.eos_id:
                            break
                    ort_text = tokenizer.decode(tokens[0].tolist())

                    status = "MATCH" if pt_text == ort_text else "DIFF"
                    print(f"    [{status}] {img_p.name:<25}: \"{ort_text}\"")

    print("\n" + "=" * 70)
    print("  ONNX Export Completed Successfully!")
    print(f"  Artifacts saved to: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
