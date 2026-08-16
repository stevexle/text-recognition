"""
Inference CLI Script for ViT-Transformer OCR Model.
Supports single image prediction, directory batch prediction, and latency benchmarking.
"""

import os
import time
import argparse
from pathlib import Path
import torch

from src.predictor import OCRPredictor
from src.utils import setup_logger


def sync_device(device: torch.device):
    """Synchronize device queue for exact latency benchmarking."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR Inference Script - Predict text and measure latency",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # 1. Predict a single image directly:
  python scripts/infer.py test/cropped/ho_va_ten.jpg

  # 2. Predict using --image flag:
  python scripts/infer.py --image path/to/image.jpg

  # 3. Predict all images in a folder and measure batch latency/FPS:
  python scripts/infer.py --image-dir test/cropped/

  # 4. Use Beam Search:
  python scripts/infer.py test/cropped/ho_va_ten.jpg --beam-size 3
        """
    )
    parser.add_argument(
        "image_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to an input image file (positional argument)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to an input image file (.jpg, .png, etc.)"
    )
    parser.add_argument(
        "--image-dir", "-d",
        type=str,
        default=None,
        help="Directory containing images to run batch inference on"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (.pt) (default: checkpoints/best_model.pt)"
    )
    parser.add_argument(
        "--vocab", "-v",
        type=str,
        default="checkpoints/vocab.json",
        help="Path to vocabulary JSON file (default: checkpoints/vocab.json)"
    )
    parser.add_argument(
        "--beam-size", "-b",
        type=int,
        default=1,
        help="Beam size for decoding (1 for greedy, >1 for beam search)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto-detected)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("InferCLI")

    # Resolve target image (positional vs flag)
    target_image = args.image_pos or args.image

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found at: {args.checkpoint}")
        return
    if not os.path.exists(args.vocab):
        logger.error(f"Vocab file not found at: {args.vocab}")
        return

    logger.info(f"Loading OCR model from '{args.checkpoint}'...")
    t_load_start = time.perf_counter()
    predictor = OCRPredictor(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device
    )
    t_load_ms = (time.perf_counter() - t_load_start) * 1000
    logger.info(f"Loaded successfully on device: {predictor.device} ({t_load_ms:.1f} ms)")

    # Case 1: Single Image
    if target_image:
        if not os.path.exists(target_image):
            logger.error(f"Image file not found: {target_image}")
            return
        
        logger.info(f"Running inference on: {target_image} (Beam Size: {args.beam_size})")
        
        sync_device(predictor.device)
        t_start = time.perf_counter()
        result = predictor.predict(target_image, beam_size=args.beam_size)
        sync_device(predictor.device)
        t_infer_ms = (time.perf_counter() - t_start) * 1000
        
        print("\n" + "=" * 55)
        print(f"Image:    {target_image}")
        print(f"Result:   {result}")
        print(f"Latency:  {t_infer_ms:.2f} ms ({1000/t_infer_ms:.1f} FPS)")
        print("=" * 55 + "\n")
        return

    # Case 2: Image Directory
    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            logger.error(f"Directory not found: {args.image_dir}")
            return
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            str(p) for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in valid_exts
        ]
        image_files.sort()

        if not image_files:
            logger.warning(f"No image files found in '{args.image_dir}'")
            return

        logger.info(f"Found {len(image_files)} images in '{args.image_dir}'. Running inference & measuring time...")
        
        predictions = []
        latencies_ms = []

        for p in image_files:
            sync_device(predictor.device)
            t_s = time.perf_counter()
            pred = predictor.predict(p, beam_size=args.beam_size)
            sync_device(predictor.device)
            lat = (time.perf_counter() - t_s) * 1000
            predictions.append(pred)
            latencies_ms.append(lat)

        total_time_ms = sum(latencies_ms)
        avg_time_ms = total_time_ms / len(image_files)
        fps = (len(image_files) / total_time_ms) * 1000

        print("\n" + "=" * 80)
        print(f"{'Image Name':<30} | {'Latency':<10} | {'Predicted Text'}")
        print("-" * 80)
        for path, lat, text in zip(image_files, latencies_ms, predictions):
            print(f"{Path(path).name:<30} | {lat:>6.2f} ms | {text}")
        print("=" * 80)
        print(f"Total Images: {len(image_files)} | Total Time: {total_time_ms:.1f} ms | Avg Latency: {avg_time_ms:.2f} ms | Speed: {fps:.1f} FPS")
        print("=" * 80 + "\n")
        return

    # Case 3: Default sample demo if available
    default_dir = "data/raw/images"
    if os.path.isdir(default_dir):
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        sample_files = [
            str(p) for p in Path(default_dir).iterdir()
            if p.suffix.lower() in valid_exts
        ][:5]
        if sample_files:
            logger.info(f"No input provided. Running demo on 5 samples from '{default_dir}'...")
            
            predictions = []
            latencies_ms = []
            for p in sample_files:
                sync_device(predictor.device)
                t_s = time.perf_counter()
                pred = predictor.predict(p, beam_size=args.beam_size)
                sync_device(predictor.device)
                lat = (time.perf_counter() - t_s) * 1000
                predictions.append(pred)
                latencies_ms.append(lat)

            total_time_ms = sum(latencies_ms)
            avg_time_ms = total_time_ms / len(sample_files)
            fps = (len(sample_files) / total_time_ms) * 1000

            print("\n" + "=" * 80)
            print(f"{'Image Name':<30} | {'Latency':<10} | {'Predicted Text'}")
            print("-" * 80)
            for path, lat, text in zip(sample_files, latencies_ms, predictions):
                print(f"{Path(path).name:<30} | {lat:>6.2f} ms | {text}")
            print("=" * 80)
            print(f"Total Samples: {len(sample_files)} | Total Time: {total_time_ms:.1f} ms | Avg: {avg_time_ms:.2f} ms | Speed: {fps:.1f} FPS")
            print("=" * 80 + "\n")
            return

    # Case 4: No inputs
    logger.info("Usage:")
    logger.info("  python scripts/infer.py <path_to_image>")
    logger.info("  python scripts/infer.py --image-dir <path_to_folder>")


if __name__ == "__main__":
    main()
