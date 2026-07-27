"""
Project Utilities: YAML Config Loader, Rich Logger, Checkpointing, Seed Setup, and Device Resolution.
"""

import os
import random
import logging
import yaml
import numpy as np
import torch
from rich.logging import RichHandler


def set_seed(seed: int = 42):
    """
    Set random seed across random, numpy, and torch for 100% reproducible experiments.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file and resolve computing device.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config dictionary with resolved parameters
    """
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at path: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set global seed if present in config
    if "seed" in config:
        set_seed(config["seed"])

    # Resolve compute device
    device_setting = config.get("device", "auto")
    if device_setting == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_setting

    config["resolved_device"] = device
    return config


def setup_logger(
    name: str = "OCR",
    log_file: str = None,
    level: int = logging.INFO,
    show_thread: bool = False,
    show_process: bool = False
) -> logging.Logger:
    """
    Setup logging handler using Rich console formatting and optional file logger.
    Supports thread and process ID logging for multi-worker DataLoaders and DDP.
    
    Args:
        name: Logger identifier name
        log_file: Path to save log text file
        level: Logging verbosity level
        show_thread: If True, include thread name in log output
        show_process: If True, include process ID in log output
        
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Rich Console Handler for colorful terminal output
    console_handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    console_handler.setLevel(level)
    
    # Custom Log Formatter including optional Process and Thread identifiers
    fmt_prefix = ""
    if show_process:
        fmt_prefix += "[PID:%(process)d] "
    if show_thread:
        fmt_prefix += "[Thread:%(threadName)s] "
        
    formatter = logging.Formatter(f"{fmt_prefix}%(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            f"[%(asctime)s] [{fmt_prefix}%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(state: dict, filepath: str):
    """
    Save model weights checkpoint to disk.
    
    Args:
        state: Checkpoint dictionary containing model state_dict, optimizer state, epoch, metrics
        filepath: Target save path
    """
    filepath = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, map_location: str = "cpu") -> dict:
    """
    Load model weights checkpoint from disk.
    
    Args:
        filepath: Path to the checkpoint file
        map_location: Target device placement
        
    Returns:
        Checkpoint state dictionary
    """
    filepath = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at path: {filepath}")

    return torch.load(filepath, map_location=map_location)


if __name__ == "__main__":
    # Test Utilities
    test_config_path = "configs/vit_config.yaml"
    cfg = load_config(test_config_path)
    logger = setup_logger(name="OCRTest", show_thread=True, show_process=True)

    logger.info(f"Loaded config successfully from {test_config_path}")
    logger.info(f"Resolved compute device: {cfg['resolved_device']}")
    logger.info(f"Model architecture name: {cfg['model']['name']}")

    # Test checkpoint saving & loading
    dummy_state = {"epoch": 5, "best_cer": 0.02, "model_state": {"w": torch.tensor([1.0, 2.0])}}
    tmp_ckpt_path = "checkpoints/test_dummy_ckpt.pt"
    save_checkpoint(dummy_state, tmp_ckpt_path)
    loaded_state = load_checkpoint(tmp_ckpt_path)
    assert loaded_state["epoch"] == 5, "Checkpoint loading verification failed!"
    if os.path.exists(tmp_ckpt_path):
        os.remove(tmp_ckpt_path)

    print("\nAll project utilities verification tests passed successfully!")
