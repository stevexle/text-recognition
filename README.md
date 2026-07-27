# Modular OCR Text Recognition

A modular deep learning framework for cropped line text recognition trained on HuggingFace dataset [`ducto489/ocr_datasets`](https://huggingface.co/datasets/ducto489/ocr_datasets/viewer).

The repository provides a clean, extensible architecture supporting multiple state-of-the-art OCR models, dataset processing pipelines, metrics, and execution engines.

---

## ⚡ Quick Start & Environment Setup

### 1. Synchronize Environment & Install Package
Using native `uv` command to install all dependencies and set up the project package in editable mode:

```bash
# Sync dependencies and build project package
uv sync
```

### 2. Download Full Dataset (For GPU Server Training)
To download and export the full HuggingFace dataset into `./data/raw` with local images and `data.csv` metadata:

```bash
uv run python scripts/download_data.py --output-dir ./data/raw
```

### 3. Run Module Verification Tests
Since the project is installed as an editable package (`text-recognition`), all modules can be executed directly from anywhere:

```bash
# Test Data Pipeline
uv run python src/data/datamodule.py

# Test Model Architecture (Hybrid ViT Encoder & Transformer Decoder)
uv run python src/models/vit_encoder.py
uv run python src/models/transformer_decoder.py
uv run python src/models/vit_transformer.py

# Test Metrics & Utilities
uv run python src/metrics.py
uv run python src/utils.py
```

### 4. Start Model Training
To start training the Hybrid ViT + Transformer model (automatically splits dataset into 80% Train, 10% Validation, and 10% Test):

```bash
uv run python scripts/train.py --config configs/vit_config.yaml
```

### 5. Evaluate Trained Model on Test Set
To evaluate the best trained model checkpoint on the unseen Test set split:

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test-csv checkpoints/test_split.csv
```

---

## 🏗️ Supported Model Architectures

### 1. Hybrid ViT + Transformer Decoder (Default)
Combines a **Convolutional Stem (Conv2D Stem)** with a **Vision Transformer (ViT) Encoder** and an **Autoregressive Transformer Decoder**.
- **Conv2D Stem**: 3-stage convolutional layers (`Conv2D + BatchNorm + ReLU`) to extract fine-grained local character strokes and accent marks.
- **ViT Encoder**: 6-layer Multi-Head Self-Attention processing 2D feature patch embeddings $[B, N=256, d_{model}=384]$.
- **Transformer Decoder**: Masked Self-Attention and Cross-Attention connecting text tokens directly to visual memory embeddings.
- **Loss Function**: Cross-Entropy Loss with label smoothing ($0.1$).

### 2. CNN + Transformer Decoder
Uses a Convolutional Neural Network (ResNet/ConvNeXt) feature extractor combined with an Autoregressive Transformer Decoder.
- **Encoder**: Deep CNN backbone outputting feature map sequences.
- **Decoder**: Transformer Decoder predicting sequence tokens.
- **Loss Function**: Cross-Entropy Loss.

### 3. CRNN (CNN + BiLSTM + CTC)
Classic sequence recognition baseline for fast parallel inference.
- **Encoder**: CNN feature extractor.
- **Sequence Modeling**: Bidirectional LSTM (BiLSTM) sequence layers.
- **Decoder & Loss**: Connectionist Temporal Classification (CTC) Loss and Greedy/Beam Search Decoding.

---

## 📁 Repository Structure

```text
text_recognition/
├── configs/               # YAML configuration files for models and datasets
│   └── vit_config.yaml    # Config for Hybrid ViT + Transformer model
├── docs/                  # Detailed architectural specs and visual diagrams (Vietnamese)
│   ├── vit_transformer_architecture.md
│   └── vit_ocr_architecture_2d.png
├── scripts/               # Command-line entrypoint scripts
│   ├── download_data.py   # Utility script to download HuggingFace datasets
│   ├── evaluate.py        # Independent evaluation script for Test set
│   └── train.py           # Training execution script
├── src/                   # Core package source code
│   ├── data/              # Tokenizer, Dataset wrapper, DataLoader, Transforms
│   │   ├── dataset.py
│   │   ├── datamodule.py
│   │   ├── tokenizer.py
│   │   └── transforms.py
│   ├── engine/            # Training, validation, and evaluation pipeline
│   │   └── trainer.py
│   ├── models/            # Modular Model implementations (Encoders, Decoders)
│   │   ├── vit_encoder.py
│   │   ├── transformer_decoder.py
│   │   └── vit_transformer.py
│   ├── metrics.py         # Evaluation metrics (CER, WER, Exact Match Accuracy)
│   └── utils.py           # Utility functions (Config loader, Logger, Checkpoint)
├── pyproject.toml         # Package & dependency specifications
└── README.md
```

---

## 📖 Documentation

Detailed architectural breakdowns, step-by-step tensor dimension transformations, and 2D schematic diagrams are available in the [`docs/`](docs/) directory:
- [docs/vit_transformer_architecture.md](docs/vit_transformer_architecture.md)
