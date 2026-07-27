"""
Command-Line Entrypoint Script for OCR Model Training.
Handles Dataset Splitting (Train/Val/Test), Vocabulary Building, DataLoader Creation,
Model Instantiation, and Trainer Engine Execution.
"""

import os
import argparse
import pandas as pd
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
    set_seed(config.get("seed", 42))
    log_dir = config.get("log_dir", "./logs")
    log_file = os.path.join(log_dir, "train.log")
    logger = setup_logger(name="TrainCLI", log_file=log_file, show_process=True,show_thread=True)

    logger.info("==================================================") 
    logger.info("   Starting OCR Training Pipeline Script          ")
    logger.info("==================================================")
    logger.info(f"Loaded configuration from: {args.config}")
    logger.info(f"Resolved compute device:   {config['resolved_device']}")

    # 3. Read Dataset CSV Metadata & Build Vocabulary
    csv_file = config["dataset"]["csv_file"]
    img_dir = config["dataset"]["img_dir"]
    text_col = config["dataset"].get("text_column", "text")

    logger.info(f"Reading dataset CSV metadata from: {csv_file}")
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    all_texts = df[text_col].tolist()

    # Build Tokenizer vocabulary from all dataset texts
    tokenizer = Tokenizer()
    tokenizer.build_vocab_from_texts(all_texts)
    vocab_path = os.path.join(config.get("output_dir", "./checkpoints"), "vocab.json")
    tokenizer.save(vocab_path)
    logger.info(f"Built vocabulary of size: {tokenizer.vocab_size} tokens -> Saved to {vocab_path}")

    # 4. Dataset Splitting (Train 80% / Val 10% / Test 10%)
    val_split = config["dataset"].get("val_split", 0.10)
    test_split = config["dataset"].get("test_split", 0.10)
    temp_ratio = val_split + test_split
    
    seed = config.get("seed", 42)
    train_df, temp_df = train_test_split(df, test_size=temp_ratio, random_state=seed)
    val_ratio_in_temp = val_split / temp_ratio
    val_df, test_df = train_test_split(temp_df, test_size=(1.0 - val_ratio_in_temp), random_state=seed)

    # Save test set CSV for offline evaluation script
    output_dir = config.get("output_dir", "./checkpoints")
    test_csv_path = os.path.join(output_dir, "test_split.csv")
    os.makedirs(output_dir, exist_ok=True)
    test_df.to_csv(test_csv_path, index=False)

    logger.info(
        f"Dataset split completed -> "
        f"Train: {len(train_df)} samples ({1.0 - temp_ratio:.0%}) | "
        f"Val: {len(val_df)} samples ({val_split:.0%}) | "
        f"Test: {len(test_df)} samples ({test_split:.0%}) [Saved to {test_csv_path}]"
    )

    # 5. Instantiate OCRDatasets and DataLoaders
    batch_size = config["dataset"].get("batch_size", 32)
    num_workers = config["dataset"].get("num_workers", 4)
    image_size = tuple(config["dataset"].get("image_size", [32, 256]))

    train_dataset = OCRDataset(train_df, img_dir, tokenizer, image_size=image_size, is_train=True)
    val_dataset = OCRDataset(val_df, img_dir, tokenizer, image_size=image_size, is_train=False)

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 6. Instantiate Model Architecture (Hybrid ViT Encoder + Transformer Decoder)
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

    # 7. Instantiate and Run Trainer Engine
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
