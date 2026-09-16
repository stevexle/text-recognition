#!/usr/bin/env bash
# ==============================================================================
# High-Performance Runner for NVIDIA TensorRT Text Recognition
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

VENV_NVIDIA=$(find "$SCRIPT_DIR/.venv" -type d -path "*/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
if [ -n "$VENV_NVIDIA" ]; then
    export LD_LIBRARY_PATH="${VENV_NVIDIA}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    exec "$SCRIPT_DIR/.venv/bin/python" tools/predict_trt.py "$@"
else
    exec uv run python tools/predict_trt.py "$@"
fi
