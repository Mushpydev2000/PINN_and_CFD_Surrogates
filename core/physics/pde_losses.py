"""
PDE Loss Functions Module
==========================

This module defines loss functions for Physics-Informed Neural Networks (PINNs).
PINNs combine:

1. Data loss: How well predictions match observations
2. PDE residual loss: How well predictions satisfy physical equations
3. Boundary condition loss: How well BCs are satisfied

The key insight: By penalizing PDE residuals, we inject physics directly
into the loss function, making the network learn physically-consistent solutions.

LOSS FUNCTION DESIGN:
=====================

Total Loss = λ_data * L_data + λ_pde * L_pde + λ_bc * L_bc

Where:
- λ_data: Weight for data-fitting (high if we have measurements)
- λ_pde: Weight for physics (always included)
- λ_bc: Weight for boundary conditions (essential)

RESIDUAL WEIGHTING STRATEGIES:
==============================

1. Equal Weighting:
   L = MSE(u_pred - u_obs) + MSE(continuity_res) + MSE(momentum_res)
   Simple but may lead to imbalance

2. Adaptive Weighting:
   Dynamically adjust weights based on loss magnitudes during training
   Prevents one loss from dominating
   
3. Curriculum Learning:
   Start with BC loss, gradually add PDE loss
   Helps network learn domain first

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- PINNLoss: Encapsulates related functionality for modularity.
- AdaptiveWeightLoss: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np


class PINNLoss:
    """
    Physics-Informed Neural Network Loss Function.
    
    Combines multiple loss components:
    1. PDE residual loss (continuity + momentum)
    2. Boundary condition loss
    3. Data fitting loss (if measurements available)
    
    This design allows the network to learn from physics equations
    even without abundant training data.
    
    Attributes:
        pde_weight: Scaling factor for PDE residual loss
        bc_weight: Scaling factor for boundary condition loss
        data_weight: Scaling factor for data fitting loss
        loss_type: 'mse', 'l1', 'huber', 'smooth_l1'
    """
    
    def __init__(
        self,
        pde_weight: float = 1.0,
        bc_weight: float = 100.0,
        data_weight: float = 1.0,
        loss_type: str = "mse"
    ) -> None:
        """
        Initialize PINN loss function.
        
        Args:
            pde_weight: Weight for PDE residual loss
                        (increase to enforce physics more strongly)
            bc_weight: Weight for boundary condition loss
                       (typically high to enforce BCs exactly)
            data_weight: Weight for data fitting loss
                        (increase if you have experimental data)
            loss_type: Type of loss function ('mse', 'l1', 'huber', 'smooth_l1')
        """
        if pde_weight < 0 or bc_weight < 0 or data_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        
        self.pde_weight = pde_weight
        self.bc_weight = bc_weight
        self.data_weight = data_weight
        self.loss_type = loss_type
        
        # Loss history for monitoring
        self.loss_history: Dict[str, list] = {
            'total': [], 'pde': [], 'bc': [], 'data': []
        }
    
    def _compute_base_loss(self, residual: Tensor) -> Tensor:
        """
        Compute base loss function.
        
        Args:
            residual: Residual tensor (should be close to 0 for satisfied equations)
        
        Returns:
            Loss value (scalar)
        """
        if self.loss_type == "mse":
            # Mean Squared Error: Most common, emphasizes large errors
            return F.mse_loss(residual, torch.zeros_like(residual))
        
        elif self.loss_type == "l1":
            # L1 Loss: Less sensitive to outliers than MSE
            return F.l1_loss(residual, torch.zeros_like(residual))
        
        elif self.loss_type == "huber":
            # Huber Loss: Smooth approximation, good for robust training
            return F.huber_loss(residual, torch.zeros_like(residual), delta=1.0)
        
        elif self.loss_type == "smooth_l1":
            # Smooth L1 Loss: Similar to Huber
            return F.smooth_l1_loss(residual, torch.zeros_like(residual))
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def pde_loss(
        self,
        continuity_residual: Tensor,
        x_momentum_residual: Tensor,
        y_momentum_residual: Tensor
    ) -> Tensor:
        """
        Compute PDE residual loss.
        
        The PDE loss penalizes violations of the governing equations:
        - Continuity equation: ∂u/∂x + ∂v/∂y = 0
        - X-momentum: ρ(∂u/∂t + u∂u/∂x + v∂u/∂y) = -∂p/∂x + μ∇²u
        - Y-momentum: ρ(∂v/∂t + u∂v/∂x + v∂v/∂y) = -∂p/∂y + μ∇²v
        
        When these residuals are ~0, the neural network's predictions
        satisfy the physics equations at the collocation points.
        
        Args:
            continuity_residual: Continuity equation residual tensor
            x_momentum_residual: X-momentum equation residual tensor
            y_momentum_residual: Y-momentum equation residual tensor
        
        Returns:
            PDE loss (scalar)
        """
        # Compute loss for each equation
        continuity_loss = self._compute_base_loss(continuity_residual)
        x_momentum_loss = self._compute_base_loss(x_momentum_residual)
        y_momentum_loss = self._compute_base_loss(y_momentum_residual)
        
        # Total PDE loss (equally weighted equations)
        total_pde_loss = continuity_loss + x_momentum_loss + y_momentum_loss
        
        return total_pde_loss
    
    def boundary_condition_loss(
        self,
        bc_residuals: Tensor
    ) -> Tensor:
        """
        Compute boundary condition loss.
        
        Boundary conditions are constraints on the domain edges:
        - Dirichlet: Solution value is prescribed
        - Neumann: Solution gradient is prescribed
        - Periodic: Solution repeats
        
        The BC loss heavily penalizes violations of these constraints,
        ensuring the network learns the correct boundary behavior.
        
        Args:
            bc_residuals: Residuals from boundary conditions
        
        Returns:
            BC loss (scalar)
        """
        if bc_residuals.numel() == 0:
            # No BC points, return zero
            return torch.tensor(0.0, device=bc_residuals.device)
        
        return self._compute_base_loss(bc_residuals)
    
    def data_loss(
        self,
        predictions: Tensor,
        observations: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute data fitting loss.
        
        If experimental measurements are available, this term ensures
        the network predictions match the observations. This is optional
        for pure physics-informed problems.
        
        Args:
            predictions: Network predictions at data points
            observations: Measured/reference values
            mask: Optional binary mask (1 = keep, 0 = ignore)
        
        Returns:
            Data loss (scalar)
        """
        if mask is not None:
            predictions = predictions[mask.bool()]
            observations = observations[mask.bool()]
        
        return self._compute_base_loss(predictions - observations)
    
    def total_loss(
        self,
        continuity_residual: Tensor,
        x_momentum_residual: Tensor,
        y_momentum_residual: Tensor,
        bc_residuals: Optional[Tensor] = None,
        predictions: Optional[Tensor] = None,
        observations: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute total PINN loss.
        
        Combines PDE, boundary condition, and data losses into a single
        scalar loss for backpropagation during training.
        
        Args:
            continuity_residual: Continuity equation residual
            x_momentum_residual: X-momentum residual
            y_momentum_residual: Y-momentum residual
            bc_residuals: Boundary condition residuals (optional)
            predictions: Network predictions at data points (optional)
            observations: Observed values (optional)
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss components for monitoring
        """
        # Compute PDE loss
        pde_loss_val = self.pde_loss(
            continuity_residual, x_momentum_residual, y_momentum_residual
        )
        
        # Compute BC loss
        if bc_residuals is not None and bc_residuals.numel() > 0:
            bc_loss_val = self.boundary_condition_loss(bc_residuals)
        else:
            bc_loss_val = torch.tensor(0.0, device=pde_loss_val.device)
        
        # Compute data loss
        if predictions is not None and observations is not None:
            data_loss_val = self.data_loss(predictions, observations)
        else:
            data_loss_val = torch.tensor(0.0, device=pde_loss_val.device)
        
        # Compute weighted total loss
        total_loss_val = (
            self.pde_weight * pde_loss_val +
            self.bc_weight * bc_loss_val +
            self.data_weight * data_loss_val
        )
        
        # Create loss dictionary for monitoring
        loss_dict = {
            'pde': pde_loss_val.detach(),
            'bc': bc_loss_val.detach(),
            'data': data_loss_val.detach(),
            'total': total_loss_val.detach()
        }
        
        # Update history
        for key, value in loss_dict.items():
            self.loss_history[key].append(value.cpu().item())
        
        return total_loss_val, loss_dict
    
    def get_loss_history(self) -> Dict[str, list]:
        """
        Get history of loss values during training.
        
        Useful for plotting convergence curves to visualize training progress.
        
        Returns:
            Dictionary with lists of loss values over time
        """
        return self.loss_history
    
    def reset_history(self) -> None:
        """Clear loss history."""
        for key in self.loss_history:
            self.loss_history[key] = []


class AdaptiveWeightLoss:
    """
    Adaptive Weighting Loss for PINN.
    
    Dynamically adjusts loss weights during training to prevent
    one loss component from dominating others. This is important
    because the magnitude of different losses may vary greatly.
    
    Strategy: Adjust weights inversely proportional to loss magnitude
    (faster-growing losses get lower weights)
    
    Motivation: Help the network learn all aspects (physics, BCs, data)
    simultaneously rather than focusing on easiest losses.
    
    Attributes:
        initial_weights: Starting weight values
        update_frequency: Update weights every N steps
    """
    
    def __init__(
        self,
        pde_weight: float = 1.0,
        bc_weight: float = 100.0,
        data_weight: float = 1.0,
        update_frequency: int = 100
    ) -> None:
        """
        Initialize adaptive weight loss.
        
        Args:
            pde_weight: Initial PDE weight
            bc_weight: Initial BC weight
            data_weight: Initial data weight
            update_frequency: Update weights every N iterations
        """
        self.initial_weights = {
            'pde': pde_weight,
            'bc': bc_weight,
            'data': data_weight
        }
        self.current_weights = self.initial_weights.copy()
        self.update_frequency = update_frequency
        self.step_count = 0
        
        # Store loss magnitudes for adaptive weighting
        self.loss_magnitudes = {
            'pde': [],
            'bc': [],
            'data': []
        }
    
    def update_weights(
        self,
        pde_loss: Tensor,
        bc_loss: Tensor,
        data_loss: Tensor
    ) -> None:
        """
        Update loss weights based on current loss magnitudes.
        
        Uses gradient-based relative weighting approach:
        weights are inversely proportional to loss gradients.
        
        Args:
            pde_loss: Current PDE loss
            bc_loss: Current BC loss
            data_loss: Current data loss
        """
        self.step_count += 1
        
        if self.step_count % self.update_frequency == 0:
            # Store loss magnitudes
            pde_mag = pde_loss.detach().cpu().item()
            bc_mag = bc_loss.detach().cpu().item()
            data_mag = data_loss.detach().cpu().item()
            
            self.loss_magnitudes['pde'].append(pde_mag)
            self.loss_magnitudes['bc'].append(bc_mag)
            self.loss_magnitudes['data'].append(data_mag)
            
            # Compute adaptive weights (inverse proportional to magnitude)
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            
            total_mag = pde_mag + bc_mag + data_mag + eps
            
            self.current_weights['pde'] = (1.0 / (pde_mag + eps)) * self.initial_weights['pde']
            self.current_weights['bc'] = (1.0 / (bc_mag + eps)) * self.initial_weights['bc']
            self.current_weights['data'] = (1.0 / (data_mag + eps)) * self.initial_weights['data']
            
            # Normalize weights
            weight_sum = sum(self.current_weights.values())
            for key in self.current_weights:
                self.current_weights[key] /= (weight_sum + eps)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.current_weights.copy()
