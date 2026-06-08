"""
Utility Module: Reproducibility Seeding
========================================

Set seeds for reproducible results across frameworks.


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

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Ensures consistent results across:
    - Python's random module
    - NumPy
    - PyTorch (CPU and GPU)
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # PyTorch deterministic behavior
    # Note: This can impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
