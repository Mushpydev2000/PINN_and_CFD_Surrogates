"""
Utility Module: Logging Configuration
======================================

Setup logging for the project.


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

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "pinn_cfd",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Configure logger with console and file handlers.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional log file path
        log_dir: Directory for log files
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / log_file
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Default logger instance
logger = setup_logger("pinn_cfd", log_file="training.log")
