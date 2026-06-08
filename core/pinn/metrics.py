"""
PINN Metrics Module
===================

Evaluation metrics for physics-informed neural networks.
Includes metrics for assessing solution quality and physical constraints.

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- MetricsAccumulator: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from typing import Tuple, Optional
import numpy as np
from torch import Tensor
import torch


def mean_squared_error(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Mean Squared Error (MSE).
    
    MSE = 1/n * Σ(y_pred - y_true)²
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        MSE value
    """
    return np.mean((predictions - targets) ** 2)


def root_mean_squared_error(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Root Mean Squared Error (RMSE).
    
    RMSE = sqrt(MSE)
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(predictions, targets))


def mean_absolute_error(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Mean Absolute Error (MAE).
    
    MAE = 1/n * Σ|y_pred - y_true|
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(predictions - targets))


def relative_error(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Relative (normalized) error.
    
    rel_error = ||y_pred - y_true|| / ||y_true||
    
    Useful for comparing errors across different scales.
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        Relative error (0 to inf)
    """
    numerator = np.linalg.norm(predictions - targets)
    denominator = np.linalg.norm(targets)
    
    if denominator < 1e-16:
        return 0.0
    
    return numerator / denominator


def r2_score(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Coefficient of determination (R² score).
    
    R² = 1 - (SS_res / SS_tot)
    
    Where:
    - SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
    - SS_tot = Σ(y_true - mean(y_true))²  (total sum of squares)
    
    Interpretation:
    - R² = 1: Perfect prediction
    - R² = 0: Model predicts mean (no better than baseline)
    - R² < 0: Model worse than baseline
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        R² score (-inf to 1)
    """
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    
    if ss_tot < 1e-16:
        return 1.0 if ss_res < 1e-16 else 0.0
    
    return 1.0 - (ss_res / ss_tot)


def max_absolute_error(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Maximum absolute error.
    
    max_ae = max(|y_pred - y_true|)
    
    Useful for detecting worst-case errors.
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        Maximum absolute error
    """
    return np.max(np.abs(predictions - targets))


def pde_residual_norm(
    residuals: np.ndarray
) -> float:
    """
    Compute norm of PDE residuals.
    
    Measures how well the PDE is satisfied:
    - residual_norm ≈ 0: PDE well-satisfied
    - residual_norm > 0: PDE not satisfied
    
    Args:
        residuals: PDE residual tensor/array
    
    Returns:
        L2 norm of residuals
    """
    return np.linalg.norm(residuals)


def boundary_condition_error(
    bc_residuals: np.ndarray
) -> Tuple[float, float]:
    """
    Compute boundary condition satisfaction.
    
    Returns both mean and maximum errors.
    
    Args:
        bc_residuals: Boundary condition residuals
    
    Returns:
        Tuple of (mean_error, max_error)
    """
    mean_error = np.mean(np.abs(bc_residuals))
    max_error = np.max(np.abs(bc_residuals))
    
    return mean_error, max_error


class MetricsAccumulator:
    """
    Accumulate metrics over batches.
    
    Useful for computing metrics over entire validation/test sets.
    
    Example:
        acc = MetricsAccumulator()
        for batch in data_loader:
            preds = model(batch)
            acc.update(preds, batch.targets)
        metrics = acc.compute()
    """
    
    def __init__(self) -> None:
        """Initialize metrics accumulator."""
        self.predictions = []
        self.targets = []
    
    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> None:
        """
        Accumulate batch data.
        
        Args:
            predictions: Batch predictions
            targets: Batch targets
        """
        self.predictions.append(predictions.flatten())
        self.targets.append(targets.flatten())
    
    def compute(self) -> dict:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with metric values
        """
        if not self.predictions:
            raise ValueError("No data accumulated")
        
        # Concatenate all batches
        preds = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        
        # Compute metrics
        return {
            'mse': mean_squared_error(preds, targets),
            'rmse': root_mean_squared_error(preds, targets),
            'mae': mean_absolute_error(preds, targets),
            'max_ae': max_absolute_error(preds, targets),
            'rel_error': relative_error(preds, targets),
            'r2': r2_score(preds, targets),
        }
    
    def reset(self) -> None:
        """Clear accumulated data."""
        self.predictions = []
        self.targets = []


def compare_metrics(
    metrics_dict: dict,
    baseline_dict: dict
) -> dict:
    """
    Compare metrics against baseline.
    
    Args:
        metrics_dict: Current metrics
        baseline_dict: Baseline metrics
    
    Returns:
        Dictionary with relative improvements
    """
    comparison = {}
    
    for key in metrics_dict:
        current = metrics_dict[key]
        baseline = baseline_dict.get(key, current)
        
        if baseline != 0:
            improvement = (baseline - current) / abs(baseline) * 100
        else:
            improvement = 0.0
        
        comparison[f'{key}_improvement_%'] = improvement
    
    return comparison
