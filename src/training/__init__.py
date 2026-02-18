"""
训练引擎模块
基于 Unsloth 的 FunctionGemma 微调实现
"""

from .trainer import FunctionGemmaTrainer
from .callbacks import WandbCallback, SampleGenerationCallback

__all__ = ["FunctionGemmaTrainer", "WandbCallback", "SampleGenerationCallback"]
