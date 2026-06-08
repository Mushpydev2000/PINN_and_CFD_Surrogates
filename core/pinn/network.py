"""
PINN Neural Network Architecture
=================================

This module defines neural network architectures for Physics-Informed
Neural Networks. The networks learn to approximate the solution to
partial differential equations (u, v, p) while satisfying physics constraints.

ARCHITECTURE DESIGN PRINCIPLES:
===============================

1. POSITIONAL ENCODING:
   Raw coordinates often cause high-frequency artifacts in neural networks.
   Solution: Use Fourier features or other positional encodings to improve
   the network's ability to learn high-frequency spatial variations.

2. ACTIVATION FUNCTIONS:
   - ReLU: Standard, but has kinks causing issues with smooth PDE solutions
   - Tanh: Smooth, naturally periodic-like, good for PDE problems
   - Sin: Fully periodic, excellent for smooth solutions
   - GELU: Smooth transition, works well in practice

3. NETWORK DEPTH:
   - Deeper networks have more expressive power
   - Risk: Gradient vanishing/explosion
   - Solution: Use skip connections (residual networks)

4. BATCH NORMALIZATION:
   - Normalizes layer outputs
   - Helps training but can interfere with physical constraints
   - Often disabled for PINNs

5. OUTPUT SCALING:
   - Outputs often need different scales (u, v typically 0-1; p can be large)
   - Solution: Learn separate scaling factors per output

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- FourierFeatures: Encapsulates related functionality for modularity.
- PINNNetwork: Encapsulates related functionality for modularity.
- ImprovedPINNNetwork: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from typing import List, Optional, Callable, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class FourierFeatures(nn.Module):
    """
    Fourier Feature Encoding (Positional Encoding).
    
    Maps raw coordinates to higher-dimensional space using sinusoidal functions.
    This helps the network learn high-frequency spatial variations.
    
    Formula:
    For input x: [sin(2^0 π x), cos(2^0 π x), sin(2^1 π x), cos(2^1 π x), ...]
    
    Motivation: Raw neural networks have bias toward learning low-frequency
    functions (Tancik et al., 2020). Fourier encoding addresses this by providing
    the network with multiple frequency components to work with.
    
    Example:
        encoder = FourierFeatures(n_freqs=8)
        x = torch.tensor([[0.5], [0.3]])  # shape: [2, 1]
        encoded = encoder(x)  # shape: [2, 16] (8 freqs * 2 (sin, cos))
    
    Attributes:
        n_freqs: Number of frequency components
        freq_matrix: Pre-computed frequency multipliers
    """
    
    def __init__(self, n_freqs: int = 8) -> None:
        """
        Initialize Fourier feature encoder.
        
        Args:
            n_freqs: Number of frequency components (higher = more detail)
        """
        super().__init__()
        self.n_freqs = n_freqs
        
        # Frequency matrix: [2^0, 2^1, 2^2, ..., 2^(n_freqs-1)]
        freq_matrix = torch.tensor(
            [2 ** i for i in range(n_freqs)],
            dtype=torch.float32
        )
        self.register_buffer('freq_matrix', freq_matrix)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input coordinates using Fourier features.
        
        Args:
            x: Input coordinates [batch_size, n_dims]
        
        Returns:
            Encoded features [batch_size, n_dims * n_freqs * 2]
        """
        # x shape: [batch_size, n_dims]
        batch_size, n_dims = x.shape
        
        # Apply each frequency to each dimension
        # Result: [batch_size, n_dims * n_freqs]
        freqs = self.freq_matrix * np.pi  # Scale by pi
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(2)  # [batch_size, n_dims, 1]
        freqs_expanded = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, n_freqs]
        
        # Compute sin and cos
        x_sin = torch.sin(x_expanded * freqs_expanded)  # [batch_size, n_dims, n_freqs]
        x_cos = torch.cos(x_expanded * freqs_expanded)  # [batch_size, n_dims, n_freqs]
        
        # Interleave sin and cos: [batch_size, n_dims, n_freqs * 2]
        x_encoded = torch.stack([x_sin, x_cos], dim=-1)  # [batch_size, n_dims, n_freqs, 2]
        x_encoded = x_encoded.reshape(batch_size, -1)  # Flatten
        
        return x_encoded


class PINNNetwork(nn.Module):
    """
    Physics-Informed Neural Network for Navier-Stokes.
    
    Maps coordinates (x, y) to fluid properties (u, v, p).
    Designed to learn smooth, differentiable functions suitable for PDE problems.
    
    Architecture:
    1. Input: (x, y) coordinates [batch_size, 2]
    2. Optional: Fourier encoding to higher dimensions
    3. Hidden layers: Multiple fully connected layers with activations
    4. Optional: Residual connections between layers
    5. Output: (u, v, p) velocities and pressure [batch_size, 3]
    
    Key Features:
    - Customizable depth and width
    - Multiple activation functions
    - Optional Fourier features
    - Optional skip connections
    - Output scaling for better numerical properties
    
    Attributes:
        use_fourier: Whether to use Fourier feature encoding
        n_fourier_freqs: Number of Fourier frequency components
        n_hidden_layers: Number of hidden layers
        n_neurons: Number of neurons per hidden layer
        activation: Activation function type
        use_residual: Whether to use skip connections
    """
    
    def __init__(
        self,
        n_inputs: int = 2,
        n_outputs: int = 3,
        n_hidden_layers: int = 6,
        n_neurons: int = 256,
        activation: str = "tanh",
        use_fourier: bool = True,
        n_fourier_freqs: int = 8,
        use_residual: bool = False,
        weight_init: str = "xavier",
        output_scaling: bool = True
    ) -> None:
        """
        Initialize PINN network.
        
        Args:
            n_inputs: Number of input dimensions (typically 2 for x, y)
            n_outputs: Number of outputs (3 for u, v, p)
            n_hidden_layers: Number of hidden layers
            n_neurons: Neurons per hidden layer
            activation: 'tanh', 'relu', 'sin', 'gelu', 'elu'
            use_fourier: Whether to use Fourier feature encoding
            n_fourier_freqs: Number of Fourier frequencies
            use_residual: Whether to use skip connections
            weight_init: 'xavier', 'he', 'normal'
            output_scaling: Whether to scale outputs
        """
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.activation_type = activation
        self.use_residual = use_residual
        self.output_scaling = output_scaling
        
        # Fourier feature encoder (optional)
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierFeatures(n_fourier_freqs)
            # Number of features after Fourier encoding: n_inputs * n_freqs * 2
            n_features = n_inputs * n_fourier_freqs * 2
        else:
            self.fourier = None
            n_features = n_inputs
        
        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sin":
            self.activation = lambda x: torch.sin(x)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers: List[nn.Module] = []
        
        # Input layer
        layers.append(nn.Linear(n_features, n_neurons))
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
        
        # Output layer
        layers.append(nn.Linear(n_neurons, n_outputs))
        
        self.layers = nn.ModuleList(layers)
        
        # Initialize weights
        self._init_weights(weight_init)
        
        # Output scaling parameters (learnable)
        if output_scaling:
            self.output_scale = nn.Parameter(torch.ones(n_outputs))
            self.output_bias = nn.Parameter(torch.zeros(n_outputs))
        
    def _init_weights(self, init_type: str) -> None:
        """
        Initialize network weights.
        
        Different initialization strategies for different activation functions:
        - Xavier: Good for tanh, sigmoid
        - He: Good for ReLU
        - Normal: General-purpose
        
        Args:
            init_type: 'xavier', 'he', 'normal'
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if init_type == "xavier":
                    nn.init.xavier_uniform_(layer.weight)
                elif init_type == "he":
                    nn.init.kaiming_uniform_(layer.weight)
                elif init_type == "normal":
                    nn.init.normal_(layer.weight, 0, 0.1)
                
                # Initialize bias to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through PINN.
        
        Args:
            x: Input coordinates [batch_size, 2] containing (x, y)
        
        Returns:
            Network output [batch_size, 3] containing (u, v, p)
        """
        # Fourier feature encoding (optional)
        if self.use_fourier:
            x = self.fourier(x)
        
        # Pass through hidden layers with skip connections
        for i, layer in enumerate(self.layers[:-1]):
            x_prev = x
            x = layer(x)
            x = self.activation(x)
            
            # Add skip connection if dimensions match and use_residual is True
            if self.use_residual and x.shape[-1] == x_prev.shape[-1]:
                x = x + x_prev
        
        # Output layer (no activation)
        output = self.layers[-1](x)
        
        # Scale output (optional)
        if self.output_scaling:
            output = output * self.output_scale + self.output_bias
        
        return output


class ImprovedPINNNetwork(nn.Module):
    """
    Improved PINN Network with Advanced Features.
    
    Adds several improvements over basic PINNNetwork:
    1. Batch normalization (optional)
    2. Dropout for regularization
    3. Multiple activation function combinations
    4. Better gradient flow
    
    Suitable for more complex problems or when dealing with
    noisy data or overfitting.
    
    Attributes:
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout probability
    """
    
    def __init__(
        self,
        n_inputs: int = 2,
        n_outputs: int = 3,
        n_hidden_layers: int = 6,
        n_neurons: int = 256,
        activation: str = "tanh",
        use_fourier: bool = True,
        n_fourier_freqs: int = 8,
        use_residual: bool = True,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.1,
        weight_init: str = "xavier"
    ) -> None:
        """
        Initialize improved PINN network.
        
        Args:
            (same as PINNNetwork, plus:)
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Probability for dropout
        """
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Fourier feature encoder
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierFeatures(n_fourier_freqs)
            n_features = n_inputs * n_fourier_freqs * 2
        else:
            self.fourier = None
            n_features = n_inputs
        
        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sin":
            self.activation = lambda x: torch.sin(x)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(n_features, n_neurons))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(n_neurons))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(n_neurons))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(n_neurons, n_outputs))
        
        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """
        Forward pass through improved PINN.
        
        Args:
            x: Input coordinates [batch_size, 2]
            training: Whether in training mode (affects dropout, batch norm)
        
        Returns:
            Network output [batch_size, 3]
        """
        # Fourier encoding
        if self.use_fourier:
            x = self.fourier(x)
        
        # Pass through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            
            if self.use_batch_norm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropouts[i](x)
        
        # Output layer
        output = self.layers[-1](x)
        
        return output
