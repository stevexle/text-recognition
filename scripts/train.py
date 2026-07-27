"""
Command-Line Entrypoint Script for OCR Model Training.
Handles Dataset Splitting (Train/Val/Test), Vocabulary Building, DataLoader Creation,
Model Instantiation, and Trainer Engine Execution.

Supports dual modes:
1. Direct HuggingFace Arrow Dataset loading (0 extra disk/inodes used).
2. Local CSV file & images loading.
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.utils import load_config, setup_logger, set_seed
from src.data.tokenizer import Tokenizer
from src.data.dataset import OCRDataset
from src.data.datamodule import build_dataloader
from src.models.vit_transformer import ViTTransformerOCR
from src.engine.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Modular OCR Text Recognition Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vit_config.yaml",
        help="Path to YAML configuration file"
    )
    return parser.parse_args()


def main():
    # 1. Parse Arguments & Load Config
    args = parse_args()
    config = load_config(args.config)
    
    # 2. Setup Seed and Logger
    seed = config.get("seed", 42)
    set_seed(seed)
    log_dir = config.get("log_dir", "./logs")
    log_file = os.path.join(log_dir, "train.log")
    logger = setup_logger(name="TrainCLI", log_file=log_file, show_process=True, show_thread=True)

    logger.info("==================================================")
    logger.info("   Starting OCR Training Pipeline Script          ")
    logger.info("==================================================")
    logger.info(f"Loaded configuration from: {args.config}")
    logger.info(f"Resolved compute device:   {config['resolved_device']}")

    # 3. Read Dataset Metadata & Build Vocabulary
    csv_file = config["dataset"].get("csv_file", "./data/raw/data.csv")
    img_dir = config["dataset"].get("img_dir", "./data/raw")
    text_col = config["dataset"].get("text_column", "text")
    dataset_name = config["dataset"].get("name", "ducto489/ocr_datasets")
    cache_dir = config["dataset"].get("cache_dir", "./data/raw")

    image_size = tuple(config["dataset"].get("image_size", [32, 256]))
    batch_size = config["dataset"].get("batch_size", 32)
    num_workers = config["dataset"].get("num_workers", 4)
    output_dir = config.get("output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")

    # Check if local CSV file exists
    if os.path.exists(csv_file):
        logger.info(f"Reading local dataset CSV metadata from: {csv_file}")
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=[text_col]).reset_index(drop=True)
        all_texts = df[text_col].tolist()

        tokenizer = Tokenizer()
        tokenizer.build_vocab_from_texts(all_texts)
        tokenizer.save(vocab_path)
        logger.info(f"Built vocabulary of size: {tokenizer.vocab_size} tokens -> Saved to {vocab_path}")

        val_split = config["dataset"].get("val_split", 0.10)
        test_split = config["dataset"].get("test_split", 0.10)
        temp_ratio = val_split + test_split

        train_df, temp_df = train_test_split(df, test_size=temp_ratio, random_state=seed)
        val_ratio_in_temp = val_split / temp_ratio
        val_df, test_df = train_test_split(temp_df, test_size=(1.0 - val_ratio_in_temp), random_state=seed)

        test_csv_path = os.path.join(output_dir, "test_split.csv")
        test_df.to_csv(test_csv_path, index=False)

        logger.info(
            f"Dataset split completed -> "
            f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
        )

        train_dataset = OCRDataset(train_df, img_dir, tokenizer, image_size=image_size, is_train=True)
        val_dataset = OCRDataset(val_df, img_dir, tokenizer, image_size=image_size, is_train=False)

    else:
        logger.info(
            f"Local CSV metadata '{csv_file}' not found. "
            f"Loading HuggingFace dataset '{dataset_name}' directly from cache: {cache_dir}"
        )
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = cache_dir
        hf_dataset = load_dataset(dataset_name, cache_dir=cache_dir)

        # Build Vocabulary directly from HuggingFace dataset splits
        sample_texts = []
        train_split_data = hf_dataset["train"]
        sample_limit = min(100000, len(train_split_data))
        logger.info(f"Scanning {sample_limit:,} HuggingFace dataset samples to build vocabulary...")

        for i in range(sample_limit):
            item = train_split_data[i]
            txt = item.get("txt") or item.get("text") or item.get("label") or ""
            if txt:
                sample_texts.append(str(txt).strip())

        tokenizer = Tokenizer()
        tokenizer.build_vocab_from_texts(sample_texts)
        tokenizer.save(vocab_path)
        logger.info(f"Built vocabulary of size: {tokenizer.vocab_size} tokens -> Saved to {vocab_path}")

        train_data = hf_dataset["train"]
        val_data = hf_dataset["validation"] if "validation" in hf_dataset else hf_dataset["test"]

        logger.info(
            f"HuggingFace dataset ready -> "
            f"Train: {len(train_data):,} samples | Val/Test: {len(val_data):,} samples"
        )

        train_dataset = OCRDataset(train_data, img_dir=None, tokenizer=tokenizer, image_size=image_size, is_train=True)
        val_dataset = OCRDataset(val_data, img_dir=None, tokenizer=tokenizer, image_size=image_size, is_train=False)

        # PyArrow memory-mapped datasets deadlock when num_workers > 0
        num_workers = 0

    # 4. Instantiate DataLoaders
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 5. Instantiate Model Architecture (Hybrid ViT Encoder + Transformer Decoder)
    model_cfg = config["model"]
    encoder_cfg = model_cfg["encoder"]
    decoder_cfg = model_cfg["decoder"]

    model = ViTTransformerOCR(
        vocab_size=tokenizer.vocab_size,
        in_channels=encoder_cfg.get("conv_stem", {}).get("in_channels", 3),
        stem_channels=encoder_cfg.get("conv_stem", {}).get("stem_channels", [64, 128, 384]),
        embed_dim=encoder_cfg.get("embed_dim", 384),
        encoder_depth=encoder_cfg.get("depth", 6),
        encoder_heads=encoder_cfg.get("num_heads", 6),
        decoder_layers=decoder_cfg.get("num_layers", 4),
        decoder_heads=decoder_cfg.get("nhead", 6),
        dim_feedforward=decoder_cfg.get("dim_feedforward", 1536),
        dropout=decoder_cfg.get("dropout", 0.1),
        image_size=image_size,
        max_seq_len=decoder_cfg.get("max_seq_len", 256),
        pad_idx=tokenizer.pad_id
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Instantiated model '{model_cfg['name']}' with {total_params:,} trainable parameters")

    # 6. Instantiate and Run Trainer Engine
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger
    )

    trainer.fit()


if __name__ == "__main__":
    main()
