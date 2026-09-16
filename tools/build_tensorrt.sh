#!/usr/bin/env bash
# ==============================================================================
# High-Performance NVIDIA TensorRT Engine Build Script for Text Recognition
# Builds Encoder and Decoder FP16 engines with dynamic shape profiles.
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}   NVIDIA TensorRT Engine Compilation Pipeline (Text Recognition)     ${NC}"
echo -e "${BLUE}======================================================================${NC}"

# 1. Verify NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}[OK] NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${RED}[ERROR] No NVIDIA GPU detected via nvidia-smi! TensorRT compilation requires an NVIDIA GPU.${NC}"
    exit 1
fi

# 2. Verify TensorRT Python package in virtual environment
if ! uv run python -c "import tensorrt; print('TensorRT Version:', tensorrt.__version__)" &> /dev/null; then
    echo -e "${YELLOW}[INFO] TensorRT not found in active venv. Synchronizing dependencies...${NC}"
    uv sync
fi

# 3. Create target output directories
mkdir -p weights/tensorrt
mkdir -p weights/onnx

# 4. Export ONNX if not already present
if [ ! -f "weights/onnx/encoder.onnx" ] || [ ! -f "weights/onnx/decoder.onnx" ]; then
    echo -e "\n${YELLOW}[INFO] Exporting Encoder and Decoder ONNX models first...${NC}"
    uv run python tools/export_onnx.py
fi

# 5. Execute Unified Python TensorRT Builder
echo -e "\n${BLUE}--- Compiling Encoder and Decoder FP16 TensorRT Engines ---${NC}"
uv run python tools/build_tensorrt.py --model all --fp16

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}  All TensorRT Engines successfully compiled!                        ${NC}"
echo -e "${GREEN}  Saved in: weights/tensorrt/                                        ${NC}"
ls -lh weights/tensorrt/
echo -e "${BLUE}======================================================================${NC}"
