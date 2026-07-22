# Modular OCR Text Recognition Project

Dự án xây dựng mô hình OCR Text Recognition huấn luyện trên bộ dữ liệu HuggingFace [`ducto489/ocr_datasets`](https://huggingface.co/datasets/ducto489/ocr_datasets/viewer).

Dự án hỗ trợ 3 kiến trúc mô hình chính:
1. **CRNN**: CNN Backbone + BiLSTM Decoder + CTC Loss
2. **CNN + Transformer**: CNN Backbone + Transformer Decoder + Cross Entropy Loss
3. **ViT + Transformer**: ViT Encoder + Transformer Decoder + Cross Entropy Loss

---

## 🛠️ Cài đặt & Quản lý Môi trường với `uv`

Yêu cầu: **Python 3.12** và **`uv`**.

```bash
# 1. Cài đặt các gói phụ thuộc và tạo môi trường ảo .venv
uv sync

# 2. Khai báo/Chạy thử nghiệm mô hình
uv run python -c "import torch; print('PyTorch Version:', torch.__version__)"
```

---

## 📁 Cấu trúc Thư mục Dự án

```text
text_recognition/
├── configs/            # File cấu hình YAML cho từng mô hình
├── src/                # Mã nguồn chính (Data, Models, Losses, Metrics, Engine)
├── scripts/            # Script khởi chạy CLI (train, evaluate, infer)
├── checkpoints/        # Nơi lưu checkpoint mô hình
├── logs/               # Nơi lưu log huấn luyện
└── pyproject.toml      # Khai báo phụ thuộc uv
```
