#!/usr/bin/env bash
# ==============================================================================
# High-Performance Runner for NVIDIA TensorRT Text Recognition
# ==============================================================================

set -e

VENV_NVIDIA=$(find "$PWD/.venv" -type d -path "*/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
if [ -n "$VENV_NVIDIA" ]; then
    export LD_LIBRARY_PATH="${VENV_NVIDIA}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec uv run python tools/predict_trt.py "$@"
