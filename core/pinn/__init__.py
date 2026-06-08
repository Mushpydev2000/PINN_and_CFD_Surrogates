"""Physics-Informed Neural Network (PINN) module."""

from .network import PINNNetwork, ImprovedPINNNetwork, FourierFeatures
from .trainer import PINNTrainer
from .inference import PINNInference
from .metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    relative_error,
    r2_score,
    max_absolute_error,
    pde_residual_norm,
    boundary_condition_error,
    MetricsAccumulator,
    compare_metrics,
)

__all__ = [
    'PINNNetwork',
    'ImprovedPINNNetwork',
    'FourierFeatures',
    'PINNTrainer',
    'PINNInference',
    'mean_squared_error',
    'root_mean_squared_error',
    'mean_absolute_error',
    'relative_error',
    'r2_score',
    'max_absolute_error',
    'pde_residual_norm',
    'boundary_condition_error',
    'MetricsAccumulator',
    'compare_metrics',
]
