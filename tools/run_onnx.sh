#!/usr/bin/env bash
# ==============================================================================
# High-Performance Runner for ONNX Runtime with NVIDIA CUDA Auto-Discovery
# Automatically locates and exports NVIDIA cublas/cudnn libraries from .venv
# Runs virtualenv python directly to prevent uv run from stripping LD_LIBRARY_PATH
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Find all nvidia library directories inside .venv
VENV_NVIDIA=$(find "$SCRIPT_DIR/.venv" -type d -path "*/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
if [ -n "$VENV_NVIDIA" ]; then
    export LD_LIBRARY_PATH="${VENV_NVIDIA}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Run via .venv/bin/python directly to ensure LD_LIBRARY_PATH is preserved for glibc
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    exec "$SCRIPT_DIR/.venv/bin/python" tools/predict_onnx.py "$@"
else
    exec uv run python tools/predict_onnx.py "$@"
fi
