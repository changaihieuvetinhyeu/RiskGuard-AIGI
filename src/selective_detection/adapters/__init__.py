"""Official detector adapters for RiskGuard-AIGI Phase 2."""

from .wavelet_detector import SAFEDetector
from .vision_transformer_detector import UnivFDDetector

__all__ = ["SAFEDetector", "UnivFDDetector"]
