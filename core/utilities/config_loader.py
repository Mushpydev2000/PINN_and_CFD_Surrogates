"""
Utility Module: Configuration Management
=========================================

Load and manage YAML configuration files.


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

import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Recursively merge configurations."""
    result = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result:
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result
