"""
Utility script to download and cache HuggingFace datasets into local project data/raw directory.
"""

import os
import argparse
from datasets import load_dataset


def download_dataset(dataset_name: str = "ducto489/ocr_datasets", output_dir: str = "./data/raw"):
    """Download dataset from HuggingFace Hub and store directly inside local output_dir."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure environment variables to restrict HuggingFace downloads inside project directory
    os.environ["HF_HOME"] = output_dir
    os.environ["HF_DATASETS_CACHE"] = output_dir
    
    print(f"Downloading dataset '{dataset_name}' into directory: {output_dir}")
    
    dataset = load_dataset(dataset_name, cache_dir=output_dir)
    
    print("\nDataset downloaded successfully! Dataset summary:")
    print(dataset)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OCR dataset from HuggingFace Hub.")
    parser.add_argument("--dataset-name", type=str, default="ducto489/ocr_datasets", help="HuggingFace dataset repository name")
    parser.add_argument("--output-dir", type=str, default="./data/raw", help="Target output directory in workspace")
    
    args = parser.parse_args()
    download_dataset(dataset_name=args.dataset_name, output_dir=args.output_dir)
