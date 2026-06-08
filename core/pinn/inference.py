"""
PINN Inference Module
=====================

This module provides inference functionality for trained PINN models.
Includes:
1. Solution evaluation on arbitrary points
2. Gradient computation (for visualization)
3. Domain-wide solution computation
4. Statistical analysis of solutions

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- PINNInference: Encapsulates related functionality for modularity.

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
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class PINNInference:
    """
    Inference interface for trained PINN models.
    
    Evaluates the solution at arbitrary points and computes
    quantities of interest.
    
    Attributes:
        model: Trained PINN network
        device: 'cuda' or 'cpu'
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """
        Initialize inference interface.
        
        Args:
            model: Trained PINN network
            device: Device to use for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def predict(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict solution (u, v, p) at given points.
        
        Args:
            x: X-coordinates [n_points] or [n_points, 1]
            y: Y-coordinates [n_points] or [n_points, 1]
        
        Returns:
            Tuple of (u, v, p) arrays [n_points]
        """
        # Reshape if needed
        x = np.atleast_1d(x).flatten()
        y = np.atleast_1d(y).flatten()
        
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        
        # Create coordinate tensor
        coords = np.column_stack([x, y])
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
        
        # Forward pass
        output = self.model(coords_tensor)
        
        # Extract components
        u = output[:, 0].cpu().numpy()
        v = output[:, 1].cpu().numpy()
        p = output[:, 2].cpu().numpy()
        
        return u, v, p
    
    @torch.no_grad()
    def predict_on_grid(
        self,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        nx: int = 100,
        ny: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict solution on a regular grid.
        
        Args:
            x_bounds: (x_min, x_max)
            y_bounds: (y_min, y_max)
            nx: Number of points in x-direction
            ny: Number of points in y-direction
        
        Returns:
            Tuple of (X, Y, u, v, p) grid arrays
        """
        # Create grid
        x_grid = np.linspace(x_bounds[0], x_bounds[1], nx)
        y_grid = np.linspace(y_bounds[0], y_bounds[1], ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Flatten for prediction
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        # Predict
        u_flat, v_flat, p_flat = self.predict(x_flat, y_flat)
        
        # Reshape to grid
        u = u_flat.reshape(X.shape)
        v = v_flat.reshape(X.shape)
        p = p_flat.reshape(X.shape)
        
        return X, Y, u, v, p
    
    @torch.no_grad()
    def compute_velocity_magnitude(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Compute velocity magnitude |V| = sqrt(u² + v²).
        
        Args:
            x: X-coordinates
            y: Y-coordinates
        
        Returns:
            Velocity magnitude array
        """
        u, v, _ = self.predict(x, y)
        return np.sqrt(u**2 + v**2)
    
    @torch.no_grad()
    def compute_vorticity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        h: float = 0.001
    ) -> np.ndarray:
        """
        Compute vorticity ω = ∂v/∂x - ∂u/∂y using finite differences.
        
        Args:
            x: X-coordinates [n_points]
            y: Y-coordinates [n_points]
            h: Finite difference step size
        
        Returns:
            Vorticity array [n_points]
        """
        # Compute velocity at points and neighbors
        u, v, _ = self.predict(x, y)
        
        # Forward differences
        u_xp, v_xp, _ = self.predict(x + h, y)
        u_yp, v_yp, _ = self.predict(x, y + h)
        
        # Compute derivatives: ∂v/∂x - ∂u/∂y
        dv_dx = (v_xp - v) / h
        du_dy = (u_yp - u) / h
        
        omega = dv_dx - du_dy
        return omega
    
    @torch.no_grad()
    def compute_strain_rate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        h: float = 0.001
    ) -> np.ndarray:
        """
        Compute strain rate tensor: S = 1/2(∇u + ∇u^T)
        
        Returns symmetric strain rate tensor components.
        
        Args:
            x: X-coordinates
            y: Y-coordinates
            h: Finite difference step size
        
        Returns:
            Strain rate components [n_points, 3] as (Sxx, Syy, Sxy)
        """
        u, v, _ = self.predict(x, y)
        
        # Forward differences
        u_xp, v_xp, _ = self.predict(x + h, y)
        u_yp, v_yp, _ = self.predict(x, y + h)
        
        # Compute gradients
        du_dx = (u_xp - u) / h
        dv_dy = (v_yp - v) / h
        du_dy = (u_yp - u) / h
        dv_dx = (v_xp - v) / h
        
        # Strain rate components
        Sxx = du_dx
        Syy = dv_dy
        Sxy = 0.5 * (du_dy + dv_dx)
        
        return np.column_stack([Sxx, Syy, Sxy])
    
    @torch.no_grad()
    def compute_q_criterion(
        self,
        x: np.ndarray,
        y: np.ndarray,
        h: float = 0.001
    ) -> np.ndarray:
        """
        Compute Q-criterion for vortex identification.
        
        Q = 1/2(Ω² - S²) where Ω is rotation rate tensor, S is strain rate
        
        Regions with Q > 0 indicate vortex cores.
        
        Args:
            x: X-coordinates
            y: Y-coordinates
            h: Finite difference step size
        
        Returns:
            Q-criterion array [n_points]
        """
        u, v, _ = self.predict(x, y)
        
        # Forward differences
        u_xp, v_xp, _ = self.predict(x + h, y)
        u_yp, v_yp, _ = self.predict(x, y + h)
        
        # Velocity gradients
        du_dx = (u_xp - u) / h
        du_dy = (u_yp - u) / h
        dv_dx = (v_xp - v) / h
        dv_dy = (v_yp - v) / h
        
        # Strain rate tensor components
        S_xx = du_dx
        S_yy = dv_dy
        S_xy = 0.5 * (du_dy + dv_dx)
        
        S_norm_sq = S_xx**2 + S_yy**2 + 2*S_xy**2
        
        # Rotation rate tensor components
        omega = 0.5 * (dv_dx - du_dy)
        
        # Q-criterion
        Q = 0.5 * (omega**2 - S_norm_sq)
        
        return Q
    
    @torch.no_grad()
    def get_solution_statistics(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> dict:
        """
        Compute statistics of solution.
        
        Args:
            x: X-coordinates
            y: Y-coordinates
        
        Returns:
            Dictionary with statistics
        """
        u, v, p = self.predict(x, y)
        
        vel_mag = np.sqrt(u**2 + v**2)
        
        return {
            'u_min': np.min(u),
            'u_max': np.max(u),
            'u_mean': np.mean(u),
            'u_std': np.std(u),
            'v_min': np.min(v),
            'v_max': np.max(v),
            'v_mean': np.mean(v),
            'v_std': np.std(v),
            'p_min': np.min(p),
            'p_max': np.max(p),
            'p_mean': np.mean(p),
            'p_std': np.std(p),
            'vel_mag_min': np.min(vel_mag),
            'vel_mag_max': np.max(vel_mag),
            'vel_mag_mean': np.mean(vel_mag),
        }
