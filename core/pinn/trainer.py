"""
PINN Trainer Module
===================

This module implements the training pipeline for Physics-Informed Neural Networks.
Handles:
1. Data sampling (collocation points, boundary points)
2. Forward pass and loss computation
3. Optimization and learning rate scheduling
4. Checkpointing and early stopping
5. Logging and tensorboard support

TRAINING STRATEGY:
==================

Sampling:
- Interior collocation points: Random/LHS sampling from domain
- Boundary points: Sampling on boundaries
- Batch composition: Mix of interior, boundary points

Loss Computation:
- PDE residual loss at interior points
- Boundary condition loss at boundary points
- Periodic loss aggregation and logging

Optimization:
- Adam optimizer with learning rate decay
- Gradient clipping to prevent explosions
- Early stopping based on validation loss

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- PINNTrainer: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from typing import Optional, Tuple, Dict, List, Any
import os
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging

from core.physics import NavierStokesEquations2D, PINNLoss, BoundaryConditionManager
from core.pinn.network import PINNNetwork


logger = logging.getLogger(__name__)


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Networks.
    
    Manages the complete training loop including:
    - Data sampling strategy
    - Loss computation
    - Optimization
    - Validation
    - Checkpointing
    - Early stopping
    
    Attributes:
        model: PINN network
        device: 'cuda' or 'cpu'
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        pde: Navier-Stokes PDE
        bc_manager: Boundary condition manager
    """
    
    def __init__(
        self,
        model: nn.Module,
        pde: NavierStokesEquations2D,
        bc_manager: BoundaryConditionManager,
        domain_bounds: Dict[str, Tuple[float, float]],
        lr: float = 0.001,
        optimizer_type: str = "adam",
        scheduler_type: str = "exponential",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ) -> None:
        """
        Initialize PINN trainer.
        
        Args:
            model: PINN network
            pde: Navier-Stokes equations
            bc_manager: Boundary condition manager
            domain_bounds: Domain boundaries for collocation sampling
            lr: Learning rate
            optimizer_type: 'adam', 'sgd'
            scheduler_type: 'exponential', 'cosine', 'none'
            device: 'cuda' or 'cpu'
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logging
        """
        self.model = model.to(device)
        self.pde = pde
        self.bc_manager = bc_manager
        self.domain_bounds = domain_bounds
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Optimizer
        if optimizer_type == "adam":
            self.optimizer = Adam(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Learning rate scheduler
        if scheduler_type == "exponential":
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9999)
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000)
        else:
            self.scheduler = None
        
        # Loss function
        self.loss_fn = PINNLoss(
            pde_weight=1.0,
            bc_weight=100.0,
            loss_type="mse"
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def sample_collocation_points(
        self,
        n_interior: int,
        n_boundary: int,
        strategy: str = "random"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample interior and boundary collocation points.

        Interior and boundary points are returned separately so PDE and BC
        losses can each use a dedicated forward pass with intact autograd.

        Returns:
            Tuple of (x_interior, y_interior, x_boundary, y_boundary)
        """
        x_min, x_max = self.domain_bounds['x']
        y_min, y_max = self.domain_bounds['y']

        if strategy == "random":
            x_interior = np.random.uniform(x_min, x_max, n_interior)
            y_interior = np.random.uniform(y_min, y_max, n_interior)

            boundaries = np.random.choice([0, 1, 2, 3], n_boundary)
            x_boundary = []
            y_boundary = []

            for b in boundaries:
                if b == 0:
                    x_boundary.append(x_min)
                    y_boundary.append(np.random.uniform(y_min, y_max))
                elif b == 1:
                    x_boundary.append(x_max)
                    y_boundary.append(np.random.uniform(y_min, y_max))
                elif b == 2:
                    x_boundary.append(np.random.uniform(x_min, x_max))
                    y_boundary.append(y_max)
                else:
                    x_boundary.append(np.random.uniform(x_min, x_max))
                    y_boundary.append(y_min)

            x_boundary = np.array(x_boundary)
            y_boundary = np.array(y_boundary)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        x_interior_t = torch.tensor(
            x_interior, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        y_interior_t = torch.tensor(
            y_interior, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        x_boundary_t = torch.tensor(
            x_boundary, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        y_boundary_t = torch.tensor(
            y_boundary, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        return x_interior_t, y_interior_t, x_boundary_t, y_boundary_t

    def train_step(
        self,
        x_interior: torch.Tensor,
        y_interior: torch.Tensor,
        x_boundary: torch.Tensor,
        y_boundary: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step with separate interior and boundary forward passes.

        Returns:
            Dictionary with loss values
        """
        self.model.train()

        # Interior forward pass for PDE residuals
        coords_interior = torch.cat([x_interior, y_interior], dim=1)
        coords_interior.requires_grad_(True)
        output_interior = self.model(coords_interior)
        u_interior = output_interior[:, 0:1]
        v_interior = output_interior[:, 1:2]
        p_interior = output_interior[:, 2:3]

        continuity_res, x_momentum_res, y_momentum_res = (
            self.pde.compute_all_residuals(
                u_interior, v_interior, p_interior, coords_interior
            )
        )

        # Boundary forward pass for BC residuals
        coords_boundary = torch.cat([x_boundary, y_boundary], dim=1)
        output_boundary = self.model(coords_boundary)
        u_boundary = output_boundary[:, 0:1]
        v_boundary = output_boundary[:, 1:2]
        p_boundary = output_boundary[:, 2:3]
        x_bnd = coords_boundary[:, 0:1]
        y_bnd = coords_boundary[:, 1:2]

        bc_residuals = self.bc_manager.compute_residuals(
            u_boundary, v_boundary, p_boundary, x_bnd, y_bnd
        )
        
        # Compute loss
        total_loss, loss_dict = self.loss_fn.total_loss(
            continuity_res, x_momentum_res, y_momentum_res,
            bc_residuals=bc_residuals
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimization step
        self.optimizer.step()
        
        return {
            'total_loss': loss_dict['total'].item(),
            'pde_loss': loss_dict['pde'].item(),
            'bc_loss': loss_dict['bc'].item(),
            'data_loss': loss_dict['data'].item()
        }
    
    def train(
        self,
        n_epochs: int,
        n_interior: int = 10000,
        n_boundary: int = 2000,
        batch_size: int = 256,
        early_stopping_patience: int = 100,
        val_freq: int = 50,
        log_freq: int = 100
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            n_epochs: Number of training epochs
            n_interior: Interior collocation points per epoch
            n_boundary: Boundary points per epoch
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            val_freq: Validation frequency (every N epochs)
            log_freq: Logging frequency (every N epochs)
        
        Returns:
            Training history dictionary
        """
        history = {'loss': [], 'pde_loss': [], 'bc_loss': []}
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            
            # Sample collocation points
            x_int, y_int, x_bnd, y_bnd = self.sample_collocation_points(
                n_interior, n_boundary
            )

            # Training step
            loss_dict = self.train_step(x_int, y_int, x_bnd, y_bnd)
            
            history['loss'].append(loss_dict['total_loss'])
            history['pde_loss'].append(loss_dict['pde_loss'])
            history['bc_loss'].append(loss_dict['bc_loss'])
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            if (epoch + 1) % log_freq == 0:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs}: "
                    f"Loss={loss_dict['total_loss']:.4e}, "
                    f"PDE={loss_dict['pde_loss']:.4e}, "
                    f"BC={loss_dict['bc_loss']:.4e}"
                )
                
                # Tensorboard logging
                self.writer.add_scalar('Loss/total', loss_dict['total_loss'], epoch)
                self.writer.add_scalar('Loss/pde', loss_dict['pde_loss'], epoch)
                self.writer.add_scalar('Loss/bc', loss_dict['bc_loss'], epoch)
            
            # Early stopping
            if loss_dict['total_loss'] < self.best_loss:
                self.best_loss = loss_dict['total_loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"with best loss {self.best_loss:.4e}"
                    )
                    break
        
        self.writer.close()
        return history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        logger.info(f"Checkpoint loaded from {path}")
