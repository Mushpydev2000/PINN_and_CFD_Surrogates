"""Device resolution helpers.

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

import torch


def resolve_device(device: str = "auto") -> str:
    """Resolve a requested device string to an available runtime device."""
    if device in ("auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"

    return device
