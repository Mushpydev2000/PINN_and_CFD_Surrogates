"""
CFD Surrogate Model Evaluator
=============================

Comprehensive evaluation and analysis of surrogate model predictions.
Includes metrics, error analysis, and prediction comparison.

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- SurrogateEvaluator: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class SurrogateEvaluator:
    """
    Comprehensive evaluator for surrogate models.
    
    Computes multiple metrics and provides analysis of model performance.
    
    Metrics:
    - MSE, RMSE, MAE: Error magnitudes
    - Relative error: Normalized error
    - R²: Coefficient of determination
    - Correlation: Pearson correlation
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to use
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation on dataset.
        
        Args:
            dataloader: PyTorch DataLoader with test data
        
        Returns:
            Dictionary with all metrics
        """
        all_predictions = []
        all_targets = []
        
        # Collect all predictions
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            predictions = self.model(batch_x)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
        
        # Concatenate
        y_pred = np.concatenate(all_predictions, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        metrics = self._compute_all_metrics(y_pred, y_true)
        
        return metrics
    
    @staticmethod
    def _compute_all_metrics(
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_pred: Predictions
            y_true: Ground truth
        
        Returns:
            Dictionary with metric values
        """
        # Ensure 1D arrays
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        
        # Error
        error = y_pred - y_true
        
        # MSE and variants
        mse = np.mean(error ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(error))
        max_ae = np.max(np.abs(error))
        
        # Relative error
        rel_error = np.linalg.norm(error) / (np.linalg.norm(y_true) + 1e-16)
        
        # R² score
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-16))
        
        # Correlation
        correlation = np.corrcoef(y_pred, y_true)[0, 1]
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs(error / (np.abs(y_true) + 1e-16)))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'max_ae': float(max_ae),
            'relative_error': float(rel_error),
            'r2': float(r2),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'mape': float(mape),
        }
    
    @torch.no_grad()
    def predict(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions on data.
        
        Args:
            x: Input data [n_samples, n_features]
        
        Returns:
            Predictions [n_samples, n_outputs]
        """
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        predictions = self.model(x_tensor)
        return predictions.cpu().numpy()
    
    @torch.no_grad()
    def get_error_distribution(
        self,
        dataloader: DataLoader
    ) -> Dict[str, np.ndarray]:
        """
        Get error distribution for analysis.
        
        Args:
            dataloader: Test DataLoader
        
        Returns:
            Dictionary with error statistics
        """
        all_errors = []
        all_rel_errors = []
        all_predictions = []
        all_targets = []
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            predictions = self.model(batch_x)
            
            error = (predictions - batch_y).cpu().numpy()
            rel_error = error / (np.abs(batch_y.cpu().numpy()) + 1e-16)
            
            all_errors.append(error.flatten())
            all_rel_errors.append(rel_error.flatten())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
        
        errors = np.concatenate(all_errors)
        rel_errors = np.concatenate(all_rel_errors)
        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()
        
        return {
            'absolute_errors': errors,
            'relative_errors': rel_errors,
            'predictions': predictions,
            'targets': targets,
            'error_mean': float(np.mean(errors)),
            'error_std': float(np.std(errors)),
            'error_min': float(np.min(errors)),
            'error_max': float(np.max(errors)),
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]) -> None:
        """Pretty print metrics."""
        logger.info("=" * 50)
        logger.info("MODEL EVALUATION METRICS")
        logger.info("=" * 50)
        
        for name, value in sorted(metrics.items()):
            if name.lower() == 'r2':
                logger.info(f"{name:.<30} {value:>10.6f}")
            else:
                logger.info(f"{name:.<30} {value:>10.6e}")
        
        logger.info("=" * 50)
    
    @staticmethod
    def get_error_percentiles(
        errors: np.ndarray,
        percentiles: list = [10, 25, 50, 75, 90, 95, 99]
    ) -> Dict[str, float]:
        """
        Compute error percentiles.
        
        Args:
            errors: Error array
            percentiles: Percentiles to compute
        
        Returns:
            Dictionary with percentile values
        """
        abs_errors = np.abs(errors)
        
        result = {}
        for p in percentiles:
            result[f'p{p}'] = float(np.percentile(abs_errors, p))
        
        return result


def create_error_report(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """
    Create comprehensive error report.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Formatted report string
    """
    evaluator = SurrogateEvaluator(model, device)
    
    # Get metrics
    metrics = evaluator.evaluate(test_loader)
    
    # Get error distribution
    error_dist = evaluator.get_error_distribution(test_loader)
    error_percentiles = evaluator.get_error_percentiles(error_dist['absolute_errors'])
    
    # Format report
    report = []
    report.append("=" * 70)
    report.append("CFD SURROGATE MODEL EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Main metrics
    report.append("PRIMARY METRICS:")
    report.append(f"  R² Score:              {metrics['r2']:>10.6f}")
    report.append(f"  RMSE:                  {metrics['rmse']:>10.6e}")
    report.append(f"  MAE:                   {metrics['mae']:>10.6e}")
    report.append(f"  Max Absolute Error:    {metrics['max_ae']:>10.6e}")
    report.append(f"  Relative Error:        {metrics['relative_error']:>10.6f}")
    report.append(f"  Correlation:           {metrics['correlation']:>10.6f}")
    report.append("")
    
    # Error distribution
    report.append("ERROR DISTRIBUTION:")
    report.append(f"  Mean Error:            {error_dist['error_mean']:>10.6e}")
    report.append(f"  Std Dev:               {error_dist['error_std']:>10.6e}")
    report.append(f"  Min Error:             {error_dist['error_min']:>10.6e}")
    report.append(f"  Max Error:             {error_dist['error_max']:>10.6e}")
    report.append("")
    
    # Error percentiles
    report.append("ERROR PERCENTILES:")
    for p, val in sorted(error_percentiles.items()):
        report.append(f"  {p:>5}:                {val:>10.6e}")
    
    report.append("=" * 70)
    
    return "\n".join(report)
