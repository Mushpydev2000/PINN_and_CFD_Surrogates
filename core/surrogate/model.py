"""
CFD Surrogate Model Architectures
==================================

Implements multiple neural network architectures for CFD surrogate modeling:

1. MLPSurrogate: Simple multi-layer perceptron
2. DeepSurrogate: Deeper network with batch normalization
3. FourierSurrogate: Uses Fourier features (placeholder)
4. DeepONetSurrogate: Operator network (placeholder)

This modular design allows easy comparison of different architectures
and extension for future methods.

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- MLPSurrogate: Encapsulates related functionality for modularity.
- DeepSurrogate: Encapsulates related functionality for modularity.
- FourierSurrogate: Encapsulates related functionality for modularity.
- DeepONetSurrogate: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor


class MLPSurrogate(nn.Module):
    """
    Multi-Layer Perceptron for CFD Surrogate.
    
    Simple baseline architecture using fully connected layers.
    Good for understanding basic surrogate model behavior.
    
    Architecture:
    Input → FC → ReLU → FC → ReLU → ... → FC → Output
    
    Attributes:
        n_inputs: Number of input features
        n_outputs: Number of output features
        n_hidden_layers: Number of hidden layers
        n_neurons: Neurons per hidden layer
    """
    
    def __init__(
        self,
        n_inputs: int = 2,
        n_outputs: int = 1,
        n_hidden_layers: int = 4,
        n_neurons: int = 128,
        activation: str = "relu",
        dropout_rate: float = 0.1
    ) -> None:
        """
        Initialize MLP surrogate.
        
        Args:
            n_inputs: Number of input features
            n_outputs: Number of output features
            n_hidden_layers: Number of hidden layers
            n_neurons: Neurons per hidden layer
            activation: 'relu', 'tanh', 'gelu'
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        layers: List[nn.Module] = []
        
        # Input layer
        layers.append(nn.Linear(n_inputs, n_neurons))
        layers.append(self.activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (no activation)
        layers.append(nn.Linear(n_neurons, n_outputs))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, n_inputs]
        
        Returns:
            Output tensor [batch_size, n_outputs]
        """
        return self.network(x)


class DeepSurrogate(nn.Module):
    """
    Deep Neural Network for CFD Surrogate.
    
    More sophisticated architecture with batch normalization and
    skip connections for better learning.
    
    Features:
    - Batch normalization for stable training
    - Skip connections (residual blocks) for gradient flow
    - More layers for complex function approximation
    
    Attributes:
        use_batch_norm: Whether to use batch normalization
        use_skip_connections: Whether to use skip connections
    """
    
    def __init__(
        self,
        n_inputs: int = 2,
        n_outputs: int = 1,
        n_hidden_layers: int = 6,
        n_neurons: int = 256,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        use_skip_connections: bool = True
    ) -> None:
        """
        Initialize deep surrogate model.
        
        Args:
            n_inputs: Number of input features
            n_outputs: Number of output features
            n_hidden_layers: Number of hidden layers
            n_neurons: Neurons per hidden layer
            activation: Activation function type
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            use_skip_connections: Whether to use skip connections
        """
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        self.use_skip_connections = use_skip_connections
        self.n_neurons = n_neurons
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # Input projection
        self.input_layer = nn.Linear(n_inputs, n_neurons)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Linear(n_neurons, n_neurons))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(n_neurons))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(n_neurons, n_outputs)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, n_inputs]
        
        Returns:
            Output tensor [batch_size, n_outputs]
        """
        # Input layer
        h = self.input_layer(x)
        h = self.activation(h)
        
        # Hidden layers with skip connections
        for i, layer in enumerate(self.hidden_layers):
            h_prev = h
            h = layer(h)
            
            if self.use_batch_norm and self.batch_norms is not None:
                h = self.batch_norms[i](h)
            
            h = self.activation(h)
            h = self.dropouts[i](h)
            
            # Skip connection if dimensions match
            if self.use_skip_connections and h.shape[-1] == h_prev.shape[-1]:
                h = h + h_prev
        
        # Output layer
        output = self.output_layer(h)
        
        return output


class FourierSurrogate(nn.Module):
    """
    Fourier Feature-based Surrogate Model.
    
    Uses Fourier feature encoding for input to help learn
    high-frequency spatial variations in CFD solutions.
    
    Placeholder for future implementation with proper
    Fourier feature engineering.
    
    Note: This is a template. Full Fourier neural operator
    implementation requires more sophisticated treatment.
    """
    
    def __init__(
        self,
        n_inputs: int = 2,
        n_outputs: int = 1,
        n_fourier_freqs: int = 8,
        n_hidden_layers: int = 4,
        n_neurons: int = 128,
        **kwargs
    ) -> None:
        """
        Initialize Fourier surrogate.
        
        Args:
            n_inputs: Number of input features
            n_outputs: Number of output features
            n_fourier_freqs: Number of Fourier frequencies
            n_hidden_layers: Number of hidden layers
            n_neurons: Neurons per hidden layer
        """
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_fourier_freqs = n_fourier_freqs
        
        # Fourier feature dimension
        n_features = n_inputs * n_fourier_freqs * 2
        
        # Simple MLP on Fourier features
        layers = []
        layers.append(nn.Linear(n_features, n_neurons))
        layers.append(nn.ReLU())
        
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(n_neurons, n_outputs))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass (basic implementation).
        
        TODO: Implement proper Fourier feature encoding and
        Fourier Neural Operator functionality.
        """
        # Placeholder: treat as regular MLP
        # In full implementation, would use Fourier features
        return self.network(x)


class DeepONetSurrogate(nn.Module):
    """
    DeepONet (Deep Operator Network) Surrogate Model.
    
    DeepONet is an architecture for learning operators
    (mappings between function spaces).
    
    Structure:
    - Branch network: Processes function input (velocity field, geometry)
    - Trunk network: Processes spatial locations (x, y coordinates)
    - Outputs: Dot product of branch and trunk outputs
    
    Advantages:
    - Can generalize across different input functions
    - Efficient for operator learning
    - Theoretical foundations in operator theory
    
    Placeholder for future implementation with proper
    DeepONet architecture.
    """
    
    def __init__(
        self,
        branch_n_inputs: int = 2,
        branch_hidden: int = 128,
        branch_depth: int = 3,
        trunk_n_inputs: int = 2,
        trunk_hidden: int = 128,
        trunk_depth: int = 3,
        output_dim: int = 1,
        **kwargs
    ) -> None:
        """
        Initialize DeepONet surrogate.
        
        Args:
            branch_n_inputs: Input dimension for branch network
            branch_hidden: Hidden layer size for branch
            branch_depth: Depth of branch network
            trunk_n_inputs: Input dimension for trunk network
            trunk_hidden: Hidden layer size for trunk
            trunk_depth: Depth of trunk network
            output_dim: Output dimension
        """
        super().__init__()
        
        # Branch network (processes function input)
        branch_layers = []
        branch_layers.append(nn.Linear(branch_n_inputs, branch_hidden))
        branch_layers.append(nn.ReLU())
        
        for _ in range(branch_depth - 1):
            branch_layers.append(nn.Linear(branch_hidden, branch_hidden))
            branch_layers.append(nn.ReLU())
        
        branch_layers.append(nn.Linear(branch_hidden, output_dim))
        self.branch = nn.Sequential(*branch_layers)
        
        # Trunk network (processes spatial locations)
        trunk_layers = []
        trunk_layers.append(nn.Linear(trunk_n_inputs, trunk_hidden))
        trunk_layers.append(nn.ReLU())
        
        for _ in range(trunk_depth - 1):
            trunk_layers.append(nn.Linear(trunk_hidden, trunk_hidden))
            trunk_layers.append(nn.ReLU())
        
        trunk_layers.append(nn.Linear(trunk_hidden, output_dim))
        self.trunk = nn.Sequential(*trunk_layers)
    
    def forward(
        self,
        x_branch: Tensor,
        x_trunk: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x_branch: Branch network input [batch_size, branch_dim]
            x_trunk: Trunk network input [batch_size, trunk_dim]
                     If None, treated as same as branch input
        
        Returns:
            Output tensor [batch_size, output_dim]
        """
        if x_trunk is None:
            x_trunk = x_branch
        
        # Branch and trunk outputs
        branch_out = self.branch(x_branch)  # [batch_size, output_dim]
        trunk_out = self.trunk(x_trunk)      # [batch_size, output_dim]
        
        # Operator output: element-wise product then sum
        output = (branch_out * trunk_out).sum(dim=1, keepdim=True)
        
        return output


# Factory function for creating surrogate models
def create_surrogate_model(
    model_type: str,
    n_inputs: int = 2,
    n_outputs: int = 1,
    **kwargs
) -> nn.Module:
    """
    Create surrogate model of specified type.
    
    Args:
        model_type: 'mlp', 'deep', 'fourier', 'deeponet'
        n_inputs: Number of input features
        n_outputs: Number of output features
        **kwargs: Additional arguments for specific model
    
    Returns:
        Neural network model
    """
    aliases = {
        "deep_nn": "deep",
        "fourier_neural_operator": "fourier",
    }
    model_type = aliases.get(model_type, model_type)

    if model_type == "mlp":
        return MLPSurrogate(n_inputs=n_inputs, n_outputs=n_outputs, **kwargs)
    elif model_type == "deep":
        return DeepSurrogate(n_inputs=n_inputs, n_outputs=n_outputs, **kwargs)
    elif model_type == "fourier":
        return FourierSurrogate(n_inputs=n_inputs, n_outputs=n_outputs, **kwargs)
    elif model_type == "deeponet":
        return DeepONetSurrogate(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
