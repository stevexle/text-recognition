"""
Evaluation Entrypoint Script for Evaluating Trained OCR Checkpoint on Test Set.
Computes CER, WER, and Exact Match Accuracy on unseen test data.
"""

import os
import argparse
import pandas as pd
import torch

from src.utils import load_checkpoint, setup_logger
from src.data.tokenizer import Tokenizer
from src.data.dataset import OCRDataset
from src.data.datamodule import build_dataloader
from src.models.builder import build_model
from src.metrics import compute_cer, compute_wer, compute_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained OCR Model on Test Set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint file (.pt)"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="checkpoints/vocab.json",
        help="Path to vocabulary JSON file"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="checkpoints/test_split.csv",
        help="Path to test split CSV metadata file"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam size for decoding (1 for greedy, >1 for beam search)"
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate():
    args = parse_args()
    logger = setup_logger(name="EvaluateCLI")

    logger.info("==================================================")
    logger.info("   Starting OCR Test Evaluation Script            ")
    logger.info("==================================================")

    # 1. Load Checkpoint and Tokenizer
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    device = torch.device(config.get("resolved_device", "cpu"))
    logger.info(f"Loaded checkpoint from: {args.checkpoint} (Epoch {ckpt.get('epoch', 'N/A')})")

    tokenizer = Tokenizer.load(args.vocab)
    logger.info(f"Loaded vocabulary from: {args.vocab} (Size: {tokenizer.vocab_size} tokens)")

    # 2. Load Test Dataset CSV
    text_col = config["dataset"].get("text_column", "text")
    img_dir = config["dataset"]["img_dir"]
    test_df = pd.read_csv(args.test_csv).dropna(subset=[text_col]).reset_index(drop=True)
    logger.info(f"Loaded test dataset: {len(test_df)} samples from {args.test_csv}")

    image_size = tuple(config["dataset"].get("image_size", [32, 256]))
    test_dataset = OCRDataset(test_df, img_dir, tokenizer, image_size=image_size, is_train=False)
    test_loader = build_dataloader(
        test_dataset,
        batch_size=config["dataset"].get("batch_size", 32),
        shuffle=False,
        num_workers=config["dataset"].get("num_workers", 4)
    )

    # 3. Instantiate Model Architecture dynamically & Load State Dict
    model = build_model(
        model_cfg=config.get("model", {"name": "vit_transformer"}),
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_id,
        image_size=image_size
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 4. Run Autoregressive Evaluation Loop
    all_predictions = []
    all_references = []
    max_label_length = config["dataset"].get("max_label_length", 256)

    logger.info(f"Running autoregressive text generation on Test Set (Beam Size: {args.beam_size}, Max Len: {max_label_length})...")
    for batch in test_loader:
        images = batch["images"].to(device)
        raw_texts = batch["texts"]

        gen_tokens = model.generate(
            images,
            max_len=max_label_length,
            sos_idx=tokenizer.sos_id,
            eos_idx=tokenizer.eos_id,
            beam_size=args.beam_size
        )

        pred_texts = [tokenizer.decode(tokens.cpu().tolist()) for tokens in gen_tokens]
        all_predictions.extend(pred_texts)
        all_references.extend(raw_texts)

    # 5. Compute Final Test Metrics
    cer = compute_cer(all_predictions, all_references)
    wer = compute_wer(all_predictions, all_references)
    acc = compute_accuracy(all_predictions, all_references)

    logger.info("==================================================")
    logger.info("              FINAL TEST RESULTS                  ")
    logger.info("==================================================")
    logger.info(f"Test Character Error Rate (CER): {cer:.4f} ({cer:.2%})")
    logger.info(f"Test Word Error Rate (WER):      {wer:.4f} ({wer:.2%})")
    logger.info(f"Test Exact Match Accuracy:       {acc:.4f} ({acc:.2%})")
    logger.info("==================================================")

    # Sample Predictions Printout
    logger.info("Sample Predictions:")
    for i in range(min(5, len(all_predictions))):
        logger.info(f"  [{i+1}] Pred: '{all_predictions[i]}' <=> Ref: '{all_references[i]}'")


if __name__ == "__main__":
    evaluate()
