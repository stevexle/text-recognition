"""
Training and Validation Engine for ViT + Transformer OCR Model.
Handles CrossEntropyLoss with Label Smoothing, AdamW Optimization, Cosine LR Scheduling,
Gradient Clipping, Metric Evaluation (CER/WER/Acc), and Model Checkpointing.
"""

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.tokenizer import Tokenizer
from src.metrics import compute_cer, compute_wer, compute_accuracy
from src.utils import setup_logger, save_checkpoint


class Trainer:
    """
    Trainer Pipeline for OCR Model Training and Evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        logger=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or setup_logger(name="Trainer")

        # 1. Device Placement
        self.device = torch.device(config.get("resolved_device", "cpu"))
        self.model.to(self.device)
        self.logger.info(f"Trainer initialized on device: {self.device}")

        # 2. Loss Function: CrossEntropyLoss ignoring <pad> token
        label_smoothing = config["train"].get("label_smoothing", 0.1)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=label_smoothing
        )

        # 3. Optimizer: AdamW
        lr = config["train"].get("learning_rate", 3e-4)
        weight_decay = config["train"].get("weight_decay", 0.01)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 4. Learning Rate Scheduler: Cosine Annealing
        epochs = config["train"].get("epochs", 50)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

        # 5. Checkpoint tracking
        self.output_dir = config.get("output_dir", "./checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)

        self.grad_clip = config["train"].get("gradient_clip_val", 1.0)
        self.use_amp = config["train"].get("use_amp", True) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.best_cer = float("inf")
        self.best_acc = 0.0

    def train_epoch(self, epoch: int) -> float:
        """
        Execute one epoch of model training.
        
        Args:
            epoch: Current epoch index (1-indexed)
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        total_epochs = self.config["train"]["epochs"]

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Train",
            leave=True,
            dynamic_ncols=True
        )

        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(self.device)
            targets_input = batch["targets_input"].to(self.device)
            targets_real = batch["targets_real"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP autocast: [B, L, vocab_size]
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images, targets_input)

                # Reshape for Loss computation: [B * L, vocab_size] vs [B * L]
                vocab_size = logits.size(-1)
                loss = self.criterion(
                    logits.view(-1, vocab_size),
                    targets_real.reshape(-1)
                )

            # Backward pass with AMP GradScaler & Gradient clipping
            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

            current_loss = loss.item()
            running_avg = total_loss / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "avg_loss": f"{running_avg:.4f}",
                "lr": f"{current_lr:.6f}"
            })

        avg_loss = total_loss / len(self.train_loader)
        elapsed = time.time() - start_time
        current_lr = self.optimizer.param_groups[0]["lr"]

        self.logger.info(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] "
            f"Train Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.2f}s"
        )
        return avg_loss

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> dict:
        """
        Execute model evaluation on validation set using autoregressive generation.
        
        Args:
            epoch: Current epoch index
            
        Returns:
            Dict containing val_loss, CER, WER, and Exact Match Accuracy
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []

        max_label_length = self.config["dataset"].get("max_label_length", 64)

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets_input = batch["targets_input"].to(self.device)
            targets_real = batch["targets_real"].to(self.device)
            raw_texts = batch["texts"]

            # Compute validation loss (Teacher Forcing)
            logits = self.model(images, targets_input)
            vocab_size = logits.size(-1)
            loss = self.criterion(
                logits.view(-1, vocab_size),
                targets_real.reshape(-1)
            )
            total_loss += loss.item()

            # Autoregressive Generation for OCR Metric computation
            gen_tokens = self.model.generate(
                images,
                max_len=max_label_length,
                sos_idx=self.tokenizer.sos_id,
                eos_idx=self.tokenizer.eos_id
            )

            # Decode token IDs to text strings
            pred_texts = [self.tokenizer.decode(tokens.cpu().tolist()) for tokens in gen_tokens]
            all_predictions.extend(pred_texts)
            all_references.extend(raw_texts)

        avg_val_loss = total_loss / len(self.val_loader)
        cer = compute_cer(all_predictions, all_references)
        wer = compute_wer(all_predictions, all_references)
        acc = compute_accuracy(all_predictions, all_references)

        self.logger.info(
            f"Validation Epoch [{epoch:02d}] -> "
            f"Val Loss: {avg_val_loss:.4f} | CER: {cer:.4f} | WER: {wer:.4f} | Acc: {acc:.4f}"
        )

        # Print sample predictions for visual verification
        if len(all_predictions) > 0:
            sample_pred = all_predictions[0]
            sample_ref = all_references[0]
            self.logger.info(f"Sample Pred: '{sample_pred}' <=> Ref: '{sample_ref}'")

        return {
            "val_loss": avg_val_loss,
            "cer": cer,
            "wer": wer,
            "acc": acc
        }

    def fit(self):
        """
        Main Training Loop across all epochs.
        """
        total_epochs = self.config["train"]["epochs"]
        self.logger.info(f"Starting training pipeline for {total_epochs} epochs...")

        for epoch in range(1, total_epochs + 1):
            # 1. Train 1 Epoch
            train_loss = self.train_epoch(epoch)

            # 2. Validate 1 Epoch
            val_metrics = self.validate_epoch(epoch)
            self.scheduler.step()

            # 3. Checkpointing logic
            val_cer = val_metrics["cer"]
            is_best = val_cer < self.best_cer
            if is_best:
                self.best_cer = val_cer
                self.best_acc = val_metrics["acc"]
                best_path = os.path.join(self.output_dir, "best_model.pt")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_cer": val_cer,
                        "val_acc": val_metrics["acc"],
                        "config": self.config
                    },
                    best_path
                )
                self.logger.info(f"🎯 Saved new best model to {best_path} (CER: {val_cer:.4f})")

            # Save latest model checkpoint
            latest_path = os.path.join(self.output_dir, "latest_model.pt")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_cer": val_cer,
                    "val_acc": val_metrics["acc"],
                    "config": self.config
                },
                latest_path
            )

        self.logger.info(
            f"Training completed! Best CER: {self.best_cer:.4f} | Best Accuracy: {self.best_acc:.4f}"
        )


if __name__ == "__main__":
    # Dry-run verification test of Trainer Engine using sample dataset
    from src.utils import load_config, setup_logger
    from src.data.tokenizer import Tokenizer
    from src.data.dataset import OCRDataset
    from src.data.datamodule import build_dataloader
    from src.models.vit_transformer import ViTTransformerOCR

    test_cfg = load_config("configs/vit_config.yaml")
    test_cfg["train"]["epochs"] = 2  # Set 2 epochs for quick dry run
    test_cfg["dataset"]["batch_size"] = 4

    # Build Tokenizer and Dataset
    sample_csv = test_cfg["dataset"]["csv_file"]
    sample_img_dir = test_cfg["dataset"]["img_dir"]

    tokenizer = Tokenizer()
    sample_ds = OCRDataset(sample_csv, sample_img_dir, tokenizer, is_train=True)
    sample_loader = build_dataloader(sample_ds, batch_size=4, shuffle=True)

    # Initialize Model
    model = ViTTransformerOCR(
        vocab_size=tokenizer.vocab_size,
        in_channels=3,
        stem_channels=[64, 128, 384],
        embed_dim=384,
        encoder_depth=2,
        encoder_heads=4,
        decoder_layers=2,
        decoder_heads=4,
        image_size=(32, 256),
        max_seq_len=256
    )

    logger = setup_logger(name="TrainerTest",show_process=True,show_thread=True)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=sample_loader,
        val_loader=sample_loader,
        config=test_cfg,
        logger=logger
    )

    print("--- Running Trainer Dry Run (2 Epochs) ---")
    trainer.fit()
    print("\nTrainer engine dry run completed successfully!")
