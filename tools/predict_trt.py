"""
Production NVIDIA TensorRT & ONNX Text Recognition Inference & Benchmarking CLI Tool.

Supports:
  - Single image, list of images, or recursive directory inference.
  - High-precision FP16 TensorRT execution or ONNX Runtime acceleration.
  - Benchmark performance profiling: Latency (Mean, P50, P90, P95, P99), Throughput (FPS).
  - Configurable repeat iterations (--repeat N) for production stress testing.
  - JSON results export.
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time
from typing import List, Optional

# Auto-detect and register NVIDIA CUDA, cuDNN, and TensorRT shared libraries on Linux
if sys.platform == "linux":
    import ctypes
    import site
    try:
        for site_pkg in site.getsitepackages():
            nvidia_dir = os.path.join(site_pkg, "nvidia")
            if os.path.isdir(nvidia_dir):
                for sub in ["cuda_runtime", "cublas", "cudnn", "cufft", "curand", "tensorrt"]:
                    lib_dir = os.path.join(nvidia_dir, sub, "lib")
                    if os.path.isdir(lib_dir):
                        if "LD_LIBRARY_PATH" in os.environ:
                            if lib_dir not in os.environ["LD_LIBRARY_PATH"]:
                                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ['LD_LIBRARY_PATH']}"
                        else:
                            os.environ["LD_LIBRARY_PATH"] = lib_dir
                        for f in sorted(os.listdir(lib_dir)):
                            if f.endswith(".so") or ".so." in f:
                                try:
                                    ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
                                except Exception:
                                    pass
            trt_dir = os.path.join(site_pkg, "tensorrt")
            if os.path.isdir(trt_dir):
                for f in sorted(os.listdir(trt_dir)):
                    if f.endswith(".so") or ".so." in f:
                        try:
                            ctypes.CDLL(os.path.join(trt_dir, f), mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass
    except Exception:
        pass

import numpy as np


def collect_images(image_arg: Optional[str], input_dir_arg: Optional[str]) -> List[str]:
    """Collect image paths from CLI arguments."""
    image_paths = []
    if image_arg:
        p = Path(image_arg)
        if not p.exists():
            raise FileNotFoundError(f"Input image not found: {image_arg}")
        image_paths.append(str(p))

    if input_dir_arg:
        p = Path(input_dir_arg)
        if not p.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir_arg}")
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(glob.glob(str(p / ext)))
            image_paths.extend(glob.glob(str(p / "**" / ext), recursive=True))

    return sorted(list(set(image_paths)))


def create_predictor(
    encoder_path: str,
    decoder_path: str,
    vocab_path: str,
    max_len: int = 64,
):
    """Instantiate TensorRT predictor, or fall back to ONNX Runtime if engine files not present."""
    is_trt = encoder_path.endswith(".engine") and decoder_path.endswith(".engine")

    if is_trt:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for TensorRT.")
            from src.predictor_trt import OCRPredictorTRT
            print(f"[INFO] Initializing Native NVIDIA TensorRT Predictor:")
            print(f"       Encoder: {encoder_path}")
            print(f"       Decoder: {decoder_path}")
            return OCRPredictorTRT(
                encoder_engine=encoder_path,
                decoder_engine=decoder_path,
                vocab_path=vocab_path,
                max_len=max_len,
            ), "tensorrt"
        except Exception as e:
            print(f"[WARNING] Failed to initialize TensorRT runner ({e}). Falling back to ONNX Runtime.")

    from src.predictor_onnx import OCRPredictorONNX
    # Auto-adjust filenames if .engine paths were provided as default but only .onnx exists
    if encoder_path.endswith(".engine") and not Path(encoder_path).exists():
        fallback_enc = encoder_path.replace(".engine", ".onnx").replace("tensorrt", "onnx")
        if Path(fallback_enc).exists():
            encoder_path = fallback_enc
    if decoder_path.endswith(".engine") and not Path(decoder_path).exists():
        fallback_dec = decoder_path.replace(".engine", ".onnx").replace("tensorrt", "onnx")
        if Path(fallback_dec).exists():
            decoder_path = fallback_dec

    print(f"[INFO] Initializing ONNX Runtime Predictor:")
    print(f"       Encoder: {encoder_path}")
    print(f"       Decoder: {decoder_path}")
    return OCRPredictorONNX(
        encoder_onnx=encoder_path,
        decoder_onnx=decoder_path,
        vocab_path=vocab_path,
        max_len=max_len,
    ), "onnx"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and Predict Text Recognition via TensorRT / ONNX"
    )
    parser.add_argument("--image", type=str, help="Path to a single cropped text image")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to folder of cropped text images (default: test/cropped if exists)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="weights/tensorrt/encoder.engine",
        help="Path to compiled encoder.engine or encoder.onnx",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="weights/tensorrt/decoder.engine",
        help="Path to compiled decoder.engine or decoder.onnx",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="checkpoints/vocab.json",
        help="Path to vocabulary JSON file (default: checkpoints/vocab.json)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size (default: 16)")
    parser.add_argument("--max-len", type=int, default=64, help="Maximum generated sequence length (default: 64)")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeat iterations for latency benchmarking (default: 1)",
    )
    parser.add_argument("--output", type=str, help="Path to export predictions as JSON")
    args = parser.parse_args()

    # Collect images
    default_dir = "test/cropped" if (not args.image and not args.input_dir and Path("test/cropped").exists()) else None
    input_dir = args.input_dir or default_dir
    images = collect_images(args.image, input_dir)

    if not images:
        print("[ERROR] No images found to process. Please provide --image or --input-dir.")
        sys.exit(1)

    print(f"\n=======================================================================")
    print(f"  ViT-Transformer Text Recognition Inference & Benchmark Engine        ")
    print(f"=======================================================================")
    print(f"Target Images: {len(images)} images")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Repeat:        {args.repeat} iterations")

    # Initialize predictor
    predictor, backend = create_predictor(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        vocab_path=args.vocab,
        max_len=args.max_len,
    )

    # Warmup
    print("\nWarming up execution pipeline...")
    warmup_subset = images[: min(len(images), args.batch_size)]
    _ = predictor.predict_batch(warmup_subset, batch_size=args.batch_size)

    # Benchmark loop
    print(f"\nExecuting inference across {args.repeat} iteration(s)...")
    latencies = []
    final_predictions = []

    for r in range(args.repeat):
        t0 = time.perf_counter()
        preds = predictor.predict_batch(images, batch_size=args.batch_size)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        if r == 0:
            final_predictions = preds

    # Print Predictions
    print(f"\n--- Recognition Results (Backend: {backend.upper()}) ---")
    results_dict = {}
    for img_path, text in zip(images, final_predictions):
        name = Path(img_path).name
        results_dict[name] = text
        print(f"  {name:30s} -> \"{text}\"")

    # Benchmark Metrics
    latencies_arr = np.array(latencies)
    total_images = len(images)
    per_img_latencies = latencies_arr / total_images
    mean_lat = np.mean(per_img_latencies)
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print(f"\n=======================================================================")
    print(f"  Performance Statistics ({args.repeat} iterations, {total_images} images/iter) ")
    print(f"=======================================================================")
    print(f"  Backend:             {backend.upper()}")
    print(f"  Mean Batch Latency:  {np.mean(latencies_arr):.2f} ms")
    print(f"  Mean Per-Image Lat:  {mean_lat:.2f} ms")
    print(f"  Median (P50) Per-Img:{np.percentile(per_img_latencies, 50):.2f} ms")
    print(f"  P90 Per-Image:       {np.percentile(per_img_latencies, 90):.2f} ms")
    print(f"  P95 Per-Image:       {np.percentile(per_img_latencies, 95):.2f} ms")
    print(f"  P99 Per-Image:       {np.percentile(per_img_latencies, 99):.2f} ms")
    print(f"  Throughput:          {fps:.2f} FPS")
    print(f"=======================================================================")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": backend,
            "total_images": total_images,
            "repeat": args.repeat,
            "mean_per_image_ms": round(float(mean_lat), 3),
            "fps": round(float(fps), 2),
            "predictions": results_dict,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nExported benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
