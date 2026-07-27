"""
Utility script to download HuggingFace OCR dataset (ducto489/ocr_datasets)
and export image-text pairs to local output directory with data.csv metadata.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


def download_and_export_dataset(
    dataset_name: str = "ducto489/ocr_datasets",
    output_dir: str = "./data/raw",
    csv_name: str = "data.csv"
):
    """
    Download HuggingFace dataset and export image files + CSV metadata locally.
    
    Args:
        dataset_name: HuggingFace dataset repository identifier
        output_dir: Target local directory for dataset storage
        csv_name: Name of generated CSV metadata file
    """
    output_dir = os.path.abspath(output_dir)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Restrict HF cache inside output_dir
    os.environ["HF_HOME"] = output_dir
    os.environ["HF_DATASETS_CACHE"] = output_dir
    
    print(f"Loading HuggingFace dataset '{dataset_name}' from cache/hub into: {output_dir}")
    hf_dataset = load_dataset(dataset_name, cache_dir=output_dir)

    records = []
    split_keys = list(hf_dataset.keys())
    print(f"Dataset splits found: {split_keys}")

    img_counter = 0
    for split in split_keys:
        split_data = hf_dataset[split]
        print(f"Processing split '{split}' ({len(split_data)} samples) | Features: {split_data.column_names}")
        for item in tqdm(split_data, desc=f"Exporting {split}"):
            image = item.get("image")
            
            # Robust text key extraction across different HF OCR dataset formats
            text = None
            for key in ["text", "label", "transcription", "ground_truth", "caption", "words"]:
                if key in item and item[key] is not None and str(item[key]).strip() != "":
                    text = str(item[key]).strip()
                    break

            if image is None or text is None:
                continue

            # Save PIL image file locally
            img_filename = f"img_{img_counter:07d}.jpg"
            img_path = os.path.join(img_dir, img_filename)
            
            # Convert RGBA/Palette to RGB if necessary
            if hasattr(image, "mode") and image.mode != "RGB":
                image = image.convert("RGB")
            image.save(img_path, format="JPEG", quality=95)

            # Store relative image path for CSV
            rel_img_path = os.path.join("images", img_filename)
            records.append({
                "image": rel_img_path,
                "text": text
            })
            img_counter += 1

    # Save CSV metadata
    csv_path = os.path.join(output_dir, csv_name)
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    print("\n==================================================")
    print(f"Full dataset successfully exported to: {output_dir}")
    print(f"   Total valid samples: {len(df):,}")
    print(f"   Images saved in:     {img_dir}")
    print(f"   CSV Metadata:        {csv_path}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and export HuggingFace OCR dataset.")
    parser.add_argument("--dataset-name", type=str, default="ducto489/ocr_datasets", help="HuggingFace dataset repository name")
    parser.add_argument("--output-dir", type=str, default="./data/raw", help="Target output directory for local storage")
    parser.add_argument("--csv-name", type=str, default="data.csv", help="Metadata CSV filename")
    
    args = parser.parse_args()
    download_and_export_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        csv_name=args.csv_name
    )
