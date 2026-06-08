"""
Visualization Module
====================

Comprehensive visualization tools for PINN solutions and surrogate
model predictions. Includes contour plots, streamlines, vector fields,
and comparison plots.

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- None (Utility/Functional module)

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px


def plot_contour(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str = "Contour Plot",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    levels: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create contour plot of scalar field.
    
    Args:
        X: X-coordinates grid
        Y: Y-coordinates grid
        Z: Scalar values (velocity, pressure, etc.)
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        levels: Number of contour levels
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create contour plot
    contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Value', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_vector_field(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    title: str = "Velocity Field",
    figsize: Tuple[int, int] = (10, 8),
    scale: float = 30.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot vector field (velocities) with arrows.
    
    Args:
        X: X-coordinates
        Y: Y-coordinates
        U: X-component of velocity
        V: Y-component of velocity
        title: Plot title
        figsize: Figure size
        scale: Arrow scale factor
        save_path: Optional path to save
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot velocity magnitude as background
    vel_mag = np.sqrt(U**2 + V**2)
    im = ax.contourf(X, Y, vel_mag, levels=20, cmap='viridis', alpha=0.6)
    
    # Quiver plot (arrows)
    # Subsample for clarity
    skip = 3
    ax.quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        U[::skip, ::skip], V[::skip, ::skip],
        scale=scale, scale_units='inches', angles='xy'
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Velocity Magnitude', fontsize=12)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_streamlines(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    title: str = "Streamlines",
    figsize: Tuple[int, int] = (10, 8),
    density: float = 1.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot streamlines showing flow pattern.
    
    Args:
        X: X-coordinates
        Y: Y-coordinates
        U: X-velocity
        V: Y-velocity
        title: Plot title
        figsize: Figure size
        density: Streamline density
        save_path: Optional path to save
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Background: velocity magnitude
    vel_mag = np.sqrt(U**2 + V**2)
    im = ax.contourf(X, Y, vel_mag, levels=20, cmap='coolwarm', alpha=0.7)
    
    # Streamlines
    strm = ax.streamplot(
        X[0, :], Y[:, 0], U, V,
        color=vel_mag,
        cmap='viridis',
        density=density,
        linewidth=1.0,
        arrowsize=1.5
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Velocity Magnitude', fontsize=12)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_prediction_comparison(
    X: np.ndarray,
    Y: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    title: str = "Prediction vs Reference",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Side-by-side comparison of prediction and reference.
    
    Args:
        X: X-coordinates
        Y: Y-coordinates
        prediction: Predicted field
        reference: Reference/true field
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Prediction
    im1 = axes[0].contourf(X, Y, prediction, levels=20, cmap='viridis')
    axes[0].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0])
    
    # Reference
    im2 = axes[1].contourf(X, Y, reference, levels=20, cmap='viridis')
    axes[1].set_title('Reference', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    error = np.abs(prediction - reference)
    im3 = axes[2].contourf(X, Y, error, levels=20, cmap='hot')
    axes[2].set_title('Absolute Error', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2])
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_loss_curves(
    train_losses: list,
    val_losses: Optional[list] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    ax.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    
    if val_losses is not None:
        ax.semilogy(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_histogram(
    errors: np.ndarray,
    title: str = "Error Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of prediction errors.
    
    Args:
        errors: Array of errors
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    abs_errors = np.abs(errors)
    
    ax.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='blue')
    
    ax.axvline(np.mean(abs_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(abs_errors):.4e}')
    ax.axvline(np.median(abs_errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(abs_errors):.4e}')
    
    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs True Values",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot of predictions vs true values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Create __init__.py content for this module
__all__ = [
    'plot_contour',
    'plot_vector_field',
    'plot_streamlines',
    'plot_prediction_comparison',
    'plot_loss_curves',
    'plot_error_histogram',
    'plot_prediction_scatter',
]
