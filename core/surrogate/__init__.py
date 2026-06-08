"""CFD Surrogate Model module."""

from .dataset import CFDDatasetGenerator, CFDDataset, create_dataloaders
from .model import MLPSurrogate, DeepSurrogate, FourierSurrogate, DeepONetSurrogate, create_surrogate_model
from .trainer import SurrogateTrainer
from .evaluator import SurrogateEvaluator, create_error_report

__all__ = [
    'CFDDatasetGenerator',
    'CFDDataset',
    'create_dataloaders',
    'MLPSurrogate',
    'DeepSurrogate',
    'FourierSurrogate',
    'DeepONetSurrogate',
    'create_surrogate_model',
    'SurrogateTrainer',
    'SurrogateEvaluator',
    'create_error_report',
]
