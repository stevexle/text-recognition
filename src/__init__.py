"""
OCR Text Recognition Package.
Provides PyTorch, ONNX Runtime, and NVIDIA TensorRT predictors for ViT-Transformer Text Recognition.
"""

from src.predictor import OCRPredictor
from src.predictor_onnx import OCRPredictorONNX

try:
    from src.predictor_trt import OCRPredictorTRT
except (ImportError, RuntimeError):
    OCRPredictorTRT = None

__version__ = "0.1.0"
__all__ = ["OCRPredictor", "OCRPredictorONNX", "OCRPredictorTRT", "__version__"]
