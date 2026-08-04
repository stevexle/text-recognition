"""
Utility script to download HuggingFace OCR dataset (ducto489/ocr_datasets)
and export image-text pairs to local output directory with data.csv metadata.

Optimized for 10M+ scale datasets: Uses CSV streaming and subfolder partitioning
to prevent Linux Inode limits and RAM out-of-memory errors.
"""

import os
import io
import csv
import argparse
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset


def download_and_export_dataset(
    dataset_name: str = "ducto489/ocr_datasets",
    output_dir: str = "./data/raw",
    csv_name: str = "data.csv",
    subfolder_size: int = 100000
):
    """
    Stream download HuggingFace dataset and export partitioned image files + CSV metadata locally.
    
    Args:
        dataset_name: HuggingFace dataset repository identifier
        output_dir: Target local directory for dataset storage
        csv_name: Name of generated CSV metadata file
        subfolder_size: Max number of image files per subfolder partition
    """
    output_dir = os.path.abspath(output_dir)
    images_base_dir = os.path.join(output_dir, "images")
    os.makedirs(images_base_dir, exist_ok=True)
    
    # Restrict HF cache inside output_dir
    os.environ["HF_HOME"] = output_dir
    os.environ["HF_DATASETS_CACHE"] = output_dir
    
    print(f"Loading HuggingFace dataset '{dataset_name}' from cache/hub into: {output_dir}")
    hf_dataset = load_dataset(dataset_name, cache_dir=output_dir)

    csv_path = os.path.join(output_dir, csv_name)
    split_keys = list(hf_dataset.keys())
    print(f"Dataset splits found: {split_keys}")

    img_counter = 0
    valid_counter = 0

    # Stream write directly to CSV line-by-line to avoid RAM OOM
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "text"])

        for split in split_keys:
            split_data = hf_dataset[split]
            print(f"Processing split '{split}' ({len(split_data):,} samples) | Features: {split_data.column_names}")
            
            for item in tqdm(split_data, desc=f"Exporting {split}"):
                # 1. Robust image extraction ('jpg', 'image', 'png', 'img', 'jpeg')
                raw_image = None
                for key in ["jpg", "image", "png", "img", "jpeg"]:
                    if key in item and item[key] is not None:
                        raw_image = item[key]
                        break

                # 2. Robust text label extraction ('txt', 'text', 'label', 'transcription', 'ground_truth')
                text = None
                for key in ["txt", "text", "label", "transcription", "ground_truth", "caption", "words"]:
                    if key in item and item[key] is not None and str(item[key]).strip() != "":
                        text = str(item[key]).strip()
                        break

                if raw_image is None or text is None:
                    continue

                # Partition images into subfolders (e.g., images/part_000/, images/part_001/)
                part_idx = img_counter // subfolder_size
                part_dir_name = f"part_{part_idx:03d}"
                part_full_dir = os.path.join(images_base_dir, part_dir_name)
                if not os.path.exists(part_full_dir):
                    os.makedirs(part_full_dir, exist_ok=True)

                img_filename = f"img_{img_counter:08d}.jpg"
                img_path = os.path.join(part_full_dir, img_filename)

                # Skip if image already exported previously
                rel_img_path = os.path.join("images", part_dir_name, img_filename)
                
                if not os.path.exists(img_path):
                    try:
                        if isinstance(raw_image, bytes):
                            image = Image.open(io.BytesIO(raw_image))
                        elif isinstance(raw_image, dict) and "bytes" in raw_image and raw_image["bytes"] is not None:
                            image = Image.open(io.BytesIO(raw_image["bytes"]))
                        elif isinstance(raw_image, dict) and "path" in raw_image and raw_image["path"] and os.path.exists(str(raw_image["path"])):
                            image = Image.open(str(raw_image["path"]))
                        elif isinstance(raw_image, str) and os.path.exists(raw_image):
                            image = Image.open(raw_image)
                        else:
                            image = raw_image

                        if not isinstance(image, Image.Image):
                            img_counter += 1
                            continue

                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        image.save(img_path, format="JPEG", quality=90)
                    except Exception:
                        img_counter += 1
                        continue

                # Stream write relative image path and text label to CSV
                writer.writerow([rel_img_path, text])
                valid_counter += 1
                img_counter += 1

                if valid_counter % 50000 == 0:
                    csv_file.flush()

    print("\n==================================================")
    print(f"Full dataset successfully exported to: {output_dir}")
    print(f"   Total valid samples exported: {valid_counter:,}")
    print(f"   Images saved in partitions:   {images_base_dir}")
    print(f"   CSV Metadata:                 {csv_path}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and export HuggingFace OCR dataset.")
    parser.add_argument("--dataset-name", type=str, default="ducto489/ocr_datasets", help="HuggingFace dataset repository name")
    parser.add_argument("--output-dir", type=str, default="./data/raw", help="Target output directory for local storage")
    parser.add_argument("--csv-name", type=str, default="data.csv", help="Metadata CSV filename")
    parser.add_argument("--subfolder-size", type=int, default=100000, help="Max images per subfolder partition")
    
    args = parser.parse_args()
    download_and_export_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        csv_name=args.csv_name,
        subfolder_size=args.subfolder_size
    )
