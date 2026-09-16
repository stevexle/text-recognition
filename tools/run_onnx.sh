#!/usr/bin/env bash
# ==============================================================================
# High-Performance Runner for ONNX Runtime with NVIDIA CUDA Auto-Discovery
# Automatically locates and exports NVIDIA cublas/cudnn libraries from .venv
# ==============================================================================

set -e

# Find all nvidia library directories inside .venv (cublas, cudnn, cuda_runtime)
VENV_NVIDIA=$(find "$PWD/.venv" -type d -path "*/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
if [ -n "$VENV_NVIDIA" ]; then
    export LD_LIBRARY_PATH="${VENV_NVIDIA}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec uv run python tools/predict_onnx.py "$@"
