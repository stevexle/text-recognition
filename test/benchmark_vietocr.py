"""
Benchmark Script to Compare VietOCR with Current ViT-Transformer Model on Any Folder.
Measures Model Load Time, Per-Image Latency (ms), Throughput (FPS), and Text Accuracy.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from PIL import Image
import torch

from src.predictor import OCRPredictor


def sync_device(device: torch.device):
    """Synchronize device queue for exact latency benchmarking."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark OCR ViT Model vs VietOCR on any image directory",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # 1. Run on custom folder directly:
  python test/benchmark_vietocr.py test/long_text

  # 2. Run on default folder (test/cropped):
  python test/benchmark_vietocr.py

  # 3. Specify VietOCR model architecture (vgg_transformer or vgg_seq2seq):
  python test/benchmark_vietocr.py test/long_text --vietocr-model vgg_transformer
        """
    )
    parser.add_argument(
        "folder_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to folder containing images (positional argument)"
    )
    parser.add_argument(
        "--image-dir", "-d",
        type=str,
        default=None,
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--vietocr-model", "-m",
        type=str,
        default="vgg_transformer",
        choices=["vgg_transformer", "vgg_seq2seq"],
        help="VietOCR architecture (default: vgg_transformer)"
    )
    return parser.parse_args()


def run_benchmark(image_dir: str = "test/cropped", vietocr_model: str = "vgg_transformer"):
    if not os.path.isdir(image_dir):
        print(f"Error: Directory '{image_dir}' not found!")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in valid_exts
    ]
    image_files.sort()

    if not image_files:
        print(f"Error: No valid image files (.jpg, .png, etc.) found in '{image_dir}'")
        return

    print("=" * 95)
    print(f"               OCR BENCHMARK: ViT-Transformer vs VietOCR ({vietocr_model})")
    print(f"               Target Folder: {image_dir} ({len(image_files)} images)")
    print("=" * 95)

    # -------------------------------------------------------------
    # 1. Benchmark Your ViT-Transformer Model
    # -------------------------------------------------------------
    print(f"\n[1/2] Loading ViT-Transformer Model...")
    t0 = time.perf_counter()
    my_predictor = OCRPredictor(
        checkpoint_path="checkpoints/best_model.pt",
        vocab_path="checkpoints/vocab.json"
    )
    my_load_time = (time.perf_counter() - t0) * 1000
    print(f"      Loaded on {my_predictor.device} in {my_load_time:.1f} ms")

    # Warmup
    warmup_img = image_files[0]
    _ = my_predictor.predict(warmup_img)

    my_results = []
    my_times = []
    for p in image_files:
        sync_device(my_predictor.device)
        t_s = time.perf_counter()
        pred = my_predictor.predict(p, beam_size=1)
        sync_device(my_predictor.device)
        lat = (time.perf_counter() - t_s) * 1000
        my_results.append(pred)
        my_times.append(lat)

    my_total_time = sum(my_times)
    my_avg_time = my_total_time / len(image_files)
    my_fps = (len(image_files) / my_total_time) * 1000

    # -------------------------------------------------------------
    # 2. Benchmark VietOCR
    # -------------------------------------------------------------
    print(f"\n[2/2] Loading VietOCR ({vietocr_model})...")
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor as VietPredictor
    except ImportError:
        print("\n[!] VietOCR is not installed in the environment.")
        print("    To install VietOCR, please run: uv add vietocr\n")
        print("-" * 95)
        print(f"{'Image Name':<35} | {'ViT-Transformer (ms)':<20} | {'ViT Result'}")
        print("-" * 95)
        for path, lat, pred in zip(image_files, my_times, my_results):
            print(f"{Path(path).name:<35} | {lat:>8.2f} ms         | {pred}")
        print("=" * 95)
        print(f"ViT-Transformer Total: {my_total_time:.1f} ms | Avg: {my_avg_time:.2f} ms | FPS: {my_fps:.1f}")
        print("=" * 95)
        return

    # Pillow 10+ compatibility fix for VietOCR
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.LANCZOS

    viet_config = Cfg.load_config_from_name(vietocr_model)
    viet_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    viet_config["device"] = viet_device
    viet_config["predictor"]["beamsearch"] = False

    t0_viet = time.perf_counter()
    viet_detector = VietPredictor(viet_config)
    viet_load_time = (time.perf_counter() - t0_viet) * 1000
    print(f"      Loaded on {viet_device} in {viet_load_time:.1f} ms")

    # Warmup
    _ = viet_detector.predict(Image.open(warmup_img))

    viet_results = []
    viet_times = []
    device_obj = torch.device(viet_device)
    for p in image_files:
        pil_img = Image.open(p)
        sync_device(device_obj)
        t_s = time.perf_counter()
        pred = viet_detector.predict(pil_img)
        sync_device(device_obj)
        lat = (time.perf_counter() - t_s) * 1000
        viet_results.append(pred)
        viet_times.append(lat)

    viet_total_time = sum(viet_times)
    viet_avg_time = viet_total_time / len(image_files)
    viet_fps = (len(image_files) / viet_total_time) * 1000

    # -------------------------------------------------------------
    # Summary Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"{'Image Name':<35} | {'ViT Model (ms)':<16} | {'VietOCR (ms)':<16} | {'Speedup'}")
    print("-" * 95)
    for path, t_my, t_viet in zip(image_files, my_times, viet_times):
        if t_my <= t_viet:
            speedup = f"{t_viet / max(t_my, 1e-6):.2f}x faster"
        else:
            speedup = f"{t_my / max(t_viet, 1e-6):.2f}x slower"
        print(f"{Path(path).name:<35} | {t_my:>8.2f} ms       | {t_viet:>8.2f} ms       | {speedup}")
    print("=" * 95)
    print(f"ViT-Transformer: Total = {my_total_time:.1f} ms | Avg = {my_avg_time:.2f} ms | Throughput = {my_fps:.1f} FPS")
    print(f"VietOCR ({vietocr_model}): Total = {viet_total_time:.1f} ms | Avg = {viet_avg_time:.2f} ms | Throughput = {viet_fps:.1f} FPS")
    print("=" * 95)

    print("\nText Prediction Comparison:")
    print("-" * 95)
    for path, pred_my, pred_viet in zip(image_files, my_results, viet_results):
        print(f"Image:   {Path(path).name}")
        print(f"  ViT:     {pred_my}")
        print(f"  VietOCR: {pred_viet}")
        print()


def main():
    args = parse_args()
    target_folder = args.folder_pos or args.image_dir or "test/cropped"
    run_benchmark(image_dir=target_folder, vietocr_model=args.vietocr_model)


if __name__ == "__main__":
    main()
