"""
Production NVIDIA TensorRT Text Recognition Inference & Benchmarking CLI Tool.

Features:
  - Sub-20ms batch inference via compiled TensorRT FP16 engines (.engine).
  - Pre-loads images into RAM to isolate pure TensorRT execution latency from disk I/O.
  - Automatic fallback to ONNX Runtime if running without CUDA / TensorRT GPU environment.
  - Comprehensive statistical profiling: Latency (Mean, P50, P90, P95, P99, Min, Max, StdDev), Throughput (FPS).
  - Configurable repeat iterations (--repeat N) for stress testing with throttled progress output.
  - Standardized JSON results export.
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple

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


def collect_images(image_arg: Optional[str], input_dir_arg: Optional[str]) -> List[str]:
    """Collect image paths from CLI arguments. Automatically handles directories passed to --image."""
    image_paths = []
    targets = [p for p in [image_arg, input_dir_arg] if p]

    for target in targets:
        p = Path(target)
        if not p.exists():
            raise FileNotFoundError(f"Input path not found: {target}")
        if p.is_dir():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
                image_paths.extend(glob.glob(str(p / ext)))
                image_paths.extend(glob.glob(str(p / "**" / ext), recursive=True))
        else:
            image_paths.append(str(p))

    return sorted(list(set(image_paths)))


def preload_images(image_paths: List[str]) -> List[Tuple[str, np.ndarray]]:
    """Pre-load images into RAM to isolate pure inference latency from disk I/O."""
    loaded = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            loaded.append((p, img))
        else:
            print(f"[WARNING] Skipping unreadable image: {p}")
    return loaded


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
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations before profiling (default: 1)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeat iterations for latency benchmarking (default: 1)",
    )
    parser.add_argument("--output", type=str, help="Path to export predictions as JSON")
    args = parser.parse_args()

    default_dir = "test/cropped" if (not args.image and not args.input_dir and Path("test/cropped").exists()) else None
    input_dir = args.input_dir or default_dir
    image_paths = collect_images(args.image, input_dir)

    if not image_paths:
        print("[ERROR] No images found to process. Please provide --image or --input-dir.")
        sys.exit(1)

    print(f"\n=======================================================================")
    print(f"  ViT-Transformer Text Recognition Inference & Benchmark Engine        ")
    print(f"=======================================================================")
    print(f"Target Images: {len(image_paths)} images")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Repeat:        {args.repeat} iterations")
    print(f"Warmup:        {args.warmup} iterations")

    # 1. Preload images into RAM
    loaded_data = preload_images(image_paths)
    if not loaded_data:
        print("[ERROR] Failed to load any valid images into memory.")
        sys.exit(1)
    paths_only = [item[0] for item in loaded_data]
    raw_images = [item[1] for item in loaded_data]
    total_images = len(raw_images)

    # 2. Initialize Predictor
    predictor, backend = create_predictor(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        vocab_path=args.vocab,
        max_len=args.max_len,
    )

    # 3. Warmup
    if args.warmup > 0:
        print(f"\nWarming up execution pipeline ({args.warmup} run(s))...")
        warmup_subset = raw_images[: min(total_images, args.batch_size)]
        for _ in range(args.warmup):
            _ = predictor.predict_batch(warmup_subset, batch_size=args.batch_size)

    # 4. Benchmark Execution Loop
    repeat_count = max(1, args.repeat)
    print(f"\nExecuting benchmark inference across {repeat_count} iteration(s)...")
    latencies: List[float] = []
    final_predictions: List[str] = []

    progress_step = max(1, repeat_count // 10)
    t_start_all = time.perf_counter()

    for r in range(repeat_count):
        t0 = time.perf_counter()
        preds = predictor.predict_batch(raw_images, batch_size=args.batch_size)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

        if r == 0:
            final_predictions = preds

        # Progress reporting for high repeat runs
        if repeat_count > 20 and ((r + 1) % progress_step == 0 or (r + 1) == repeat_count):
            print(f"  Progress: [{r + 1}/{repeat_count}] iterations completed...")

    t_total_all = time.perf_counter() - t_start_all

    # 5. Print Recognition Results
    print(f"\n--- Recognition Results (Backend: {backend.upper()}) ---")
    results_dict = {}
    for img_p, text in zip(paths_only, final_predictions):
        name = Path(img_p).name
        results_dict[name] = text
        print(f"  {name:30s} -> \"{text}\"")

    # 6. Statistical Metrics Calculation
    latencies_arr = np.array(latencies)
    per_img_latencies = latencies_arr / total_images
    mean_lat = float(np.mean(per_img_latencies))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print(f"\n=======================================================================")
    print(f"  Performance Statistics ({repeat_count} iterations, {total_images} images/iter) ")
    print(f"=======================================================================")
    print(f"  Backend:             {backend.upper()}")
    print(f"  Total Runtime:       {t_total_all:.3f} s")
    print(f"  Mean Batch Latency:  {np.mean(latencies_arr):.2f} ms")
    print(f"  Mean Per-Image Lat:  {mean_lat:.2f} ms")
    print(f"  Median (P50) Per-Img:{np.percentile(per_img_latencies, 50):.2f} ms")
    print(f"  P90 Per-Image:       {np.percentile(per_img_latencies, 90):.2f} ms")
    print(f"  P95 Per-Image:       {np.percentile(per_img_latencies, 95):.2f} ms")
    print(f"  P99 Per-Image:       {np.percentile(per_img_latencies, 99):.2f} ms")
    print(f"  Min / Max Per-Image: {np.min(per_img_latencies):.2f} ms / {np.max(per_img_latencies):.2f} ms")
    print(f"  StdDev Per-Image:    {np.std(per_img_latencies):.2f} ms")
    print(f"  Throughput:          {fps:.2f} FPS")
    print(f"=======================================================================")

    # 7. JSON Export
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": backend,
            "total_images": total_images,
            "repeat": repeat_count,
            "mean_batch_ms": round(float(np.mean(latencies_arr)), 3),
            "mean_per_image_ms": round(mean_lat, 3),
            "p50_per_image_ms": round(float(np.percentile(per_img_latencies, 50)), 3),
            "p90_per_image_ms": round(float(np.percentile(per_img_latencies, 90)), 3),
            "p95_per_image_ms": round(float(np.percentile(per_img_latencies, 95)), 3),
            "p99_per_image_ms": round(float(np.percentile(per_img_latencies, 99)), 3),
            "fps": round(float(fps), 2),
            "predictions": results_dict,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nExported benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
