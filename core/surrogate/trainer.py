"""
CFD Surrogate Model Trainer
===========================

Training pipeline for CFD surrogate models including:
- Standard supervised learning
- Early stopping and checkpointing
- Learning rate scheduling
- TensorBoard logging
- Model evaluation

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- SurrogateTrainer: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from typing import Tuple, Dict, List, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SurrogateTrainer:
    """
    Trainer for CFD surrogate models.
    
    Handles full training pipeline with validation, checkpointing,
    and early stopping.
    
    Attributes:
        model: Neural network surrogate
        device: 'cuda' or 'cpu'
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 0.001,
        optimizer_type: str = "adam",
        scheduler_type: str = "exponential",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ) -> None:
        """
        Initialize surrogate trainer.
        
        Args:
            model: Neural network model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            lr: Learning rate
            optimizer_type: 'adam', 'sgd'
            scheduler_type: 'exponential', 'plateau', 'none'
            device: 'cuda' or 'cpu'
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Optimizer
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Scheduler
        if scheduler_type == "exponential":
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9995)
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=50, verbose=True
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_losses = []
        self.val_losses = []
    
    def train_step(self) -> float:
        """
        Single training epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            predictions = self.model(batch_x)
            
            # Loss
            loss = self.loss_fn(predictions, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimization step
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def validate(self) -> float:
        """
        Validation phase.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_x)
                
                # Loss
                loss = self.loss_fn(predictions, batch_y)
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def train(
        self,
        n_epochs: int = 1000,
        early_stopping_patience: int = 100,
        log_freq: int = 50
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            n_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            log_freq: Logging frequency (every N epochs)
        
        Returns:
            Training history
        """
        logger.info("Starting training...")
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            
            # Training step
            train_loss = self.train_step()
            self.training_losses.append(train_loss)
            
            # Validation step
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            if (epoch + 1) % log_freq == 0:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {train_loss:.4e} | "
                    f"Val Loss: {val_loss:.4e}"
                )
                
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                
                # Learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} with best val loss: "
                        f"{self.best_val_loss:.4e}"
                    )
                    break
        
        self.writer.close()
        
        return {
            'train_losses': self.training_losses,
            'val_losses': self.val_losses
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Checkpoint loaded from {path}")
