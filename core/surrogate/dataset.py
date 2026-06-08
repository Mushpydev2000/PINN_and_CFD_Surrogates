"""
CFD Surrogate Model Dataset Module
===================================

Generates and manages synthetic CFD simulation data for training
surrogate models. The dataset includes flow parameters and their
corresponding aerodynamic coefficients (drag, lift, pressure).

SYNTHETIC DATA GENERATION STRATEGY:
===================================

Physics-based generation using:
1. Reynolds number variation (10 to 10000)
2. Velocity variation (0.1 to 10 m/s)
3. Empirical correlations from fluid mechanics
4. Stochastic noise to simulate measurement uncertainty

Training surrogates on synthetic data allows:
- Unlimited dataset generation
- Controlled experiments
- Understanding model limitations
- Future validation with real CFD

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- CFDDatasetGenerator: Encapsulates related functionality for modularity.
- CFDDataset: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os


class CFDDatasetGenerator:
    """
    Generate synthetic CFD simulation data.
    
    Creates datasets of flow conditions and corresponding aerodynamic
    coefficients using physics-based correlations.
    
    Attributes:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_samples: int = 5000,
        seed: int = 42
    ) -> None:
        """
        Initialize dataset generator.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed
        """
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
    
    def generate_flow_conditions(self) -> Dict[str, np.ndarray]:
        """
        Generate random flow conditions.
        
        Returns:
            Dictionary with flow parameters:
            - 'velocity': Freestream velocity [m/s]
            - 'reynolds_number': Reynolds number
            - 'angle_of_attack': Angle of attack [degrees]
            - 'mach_number': Mach number
        """
        data = {
            # Velocity: uniform between 0.1 and 20 m/s
            'velocity': np.random.uniform(0.1, 20.0, self.n_samples),
            
            # Reynolds number: log-uniform between 100 and 1e6
            'reynolds_number': np.logspace(2, 6, self.n_samples),
            
            # Angle of attack: uniform between -180 and 180 degrees
            'angle_of_attack': np.random.uniform(-180, 180, self.n_samples),
            
            # Mach number: uniform between 0.1 and 0.8 (subsonic)
            'mach_number': np.random.uniform(0.1, 0.8, self.n_samples),
        }
        
        return data
    
    def cylinder_drag_coefficient(
        self,
        reynolds_number: np.ndarray
    ) -> np.ndarray:
        """
        Compute drag coefficient for cylinder using empirical correlations.
        
        Based on experimental data for flow over a circular cylinder.
        Physics: Drag depends on flow regime determined by Reynolds number.
        
        Args:
            reynolds_number: Reynolds number values
        
        Returns:
            Drag coefficient values
        """
        cd = np.zeros_like(reynolds_number)
        
        # Creeping flow regime (Re < 1)
        mask1 = reynolds_number < 1
        cd[mask1] = 24.0 / reynolds_number[mask1]
        
        # Laminar regime (1 < Re < 1000)
        mask2 = (reynolds_number >= 1) & (reynolds_number < 1000)
        cd[mask2] = 24.0 / reynolds_number[mask2] + 4.0 / np.sqrt(reynolds_number[mask2]) + 0.4
        
        # Transition regime (1000 < Re < 10000)
        mask3 = (reynolds_number >= 1000) & (reynolds_number < 10000)
        cd[mask3] = 0.4 + 2.0 / np.sqrt(reynolds_number[mask3])
        
        # Turbulent regime (Re > 10000)
        mask4 = reynolds_number >= 10000
        cd[mask4] = 1.0 + 0.1 * np.exp(-reynolds_number[mask4] / 50000)
        
        return cd
    
    def airfoil_lift_coefficient(
        self,
        angle_of_attack: np.ndarray,
        reynolds_number: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute lift coefficient for airfoil.
        
        Based on thin airfoil theory with viscous corrections.
        
        Args:
            angle_of_attack: Angle of attack [degrees]
            reynolds_number: Optional Reynolds number for viscous corrections
        
        Returns:
            Lift coefficient values
        """
        # Convert to radians
        alpha_rad = np.radians(angle_of_attack)
        
        # Thin airfoil theory: CL = 2π * sin(α)
        # Valid for small angles
        cl = 2.0 * np.pi * np.sin(alpha_rad)
        
        # Stall correction for large angles
        stall_angle = 15.0  # degrees
        stall_mask = np.abs(angle_of_attack) > stall_angle
        
        # Post-stall behavior (empirical)
        cl[stall_mask] = np.sign(angle_of_attack[stall_mask]) * (
            1.2 + 0.5 * np.cos(np.radians(angle_of_attack[stall_mask] - stall_angle))
        )
        
        return cl
    
    def compressibility_correction(
        self,
        mach_number: np.ndarray,
        coefficient: np.ndarray
    ) -> np.ndarray:
        """
        Apply Lachmann rule for compressibility correction.
        
        Prandtl-Mach correction: C_comp = C_incomp / sqrt(1 - M²)
        
        Args:
            mach_number: Mach numbers
            coefficient: Incompressible coefficients
        
        Returns:
            Corrected coefficients for compressible flow
        """
        # Avoid singularity at M = 1
        mach_sq = mach_number ** 2
        mach_sq = np.clip(mach_sq, 0, 0.99)
        
        correction_factor = 1.0 / np.sqrt(1.0 - mach_sq)
        
        return coefficient * correction_factor
    
    def generate_aerodynamic_coefficients(
        self,
        flow_conditions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Generate aerodynamic coefficients for flow conditions.
        
        Args:
            flow_conditions: Dictionary with flow parameters
        
        Returns:
            Dictionary with aerodynamic coefficients:
            - 'drag_coefficient'
            - 'lift_coefficient'
            - 'pressure_coefficient'
        """
        re = flow_conditions['reynolds_number']
        aoa = flow_conditions['angle_of_attack']
        m = flow_conditions['mach_number']
        
        # Base drag coefficient (Reynolds dependent)
        cd_base = self.cylinder_drag_coefficient(re)
        
        # Base lift coefficient (AoA dependent)
        cl_base = self.airfoil_lift_coefficient(aoa, re)
        
        # Apply compressibility corrections
        cd = self.compressibility_correction(m, cd_base)
        cl = self.compressibility_correction(m, cl_base)
        
        # Pressure coefficient (derived quantity)
        cp = 2.0 / (1.4 * m**2) * ((2.0 / (1.4 + 1)) * (1.0 + 0.5 * m**2) ** ((1.4 + 1) / (1.4 - 1)) * \
                                     ((1.4 - 1) * m**2 / 2.0 + 1.0) ** (-(1.4 + 1) / (1.4 - 1)) - 1.0)
        
        # Add noise to simulate measurement uncertainty
        noise_level = 0.02  # 2% noise
        cd_noise = cd * (1.0 + np.random.normal(0, noise_level, len(cd)))
        cl_noise = cl * (1.0 + np.random.normal(0, noise_level, len(cl)))
        cp_noise = cp * (1.0 + np.random.normal(0, noise_level, len(cp)))
        
        return {
            'drag_coefficient': cd_noise,
            'lift_coefficient': cl_noise,
            'pressure_coefficient': cp_noise,
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete CFD dataset.
        
        Returns:
            Pandas DataFrame with flow conditions and coefficients
        """
        # Generate flow conditions
        flow_cond = self.generate_flow_conditions()
        
        # Generate aerodynamic coefficients
        aero_coeff = self.generate_aerodynamic_coefficients(flow_cond)
        
        # Combine into DataFrame
        data = {
            'velocity': flow_cond['velocity'],
            'reynolds_number': flow_cond['reynolds_number'],
            'angle_of_attack': flow_cond['angle_of_attack'],
            'mach_number': flow_cond['mach_number'],
            'drag_coefficient': aero_coeff['drag_coefficient'],
            'lift_coefficient': aero_coeff['lift_coefficient'],
            'pressure_coefficient': aero_coeff['pressure_coefficient'],
        }
        
        df = pd.DataFrame(data)
        return df
    
    def save_dataset(self, filepath: str) -> None:
        """
        Generate and save dataset to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        df = self.generate_dataset()
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")


class CFDDataset(Dataset):
    """
    PyTorch Dataset for CFD surrogate model training.
    
    Loads CFD data and provides batches for model training.
    Handles data normalization and train/test splitting.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        input_features: list,
        output_features: list,
        normalize: bool = True
    ) -> None:
        """
        Initialize CFD dataset.
        
        Args:
            data: Pandas DataFrame with CFD data
            input_features: List of input feature names
            output_features: List of output feature names
            normalize: Whether to normalize data
        """
        self.data = data
        self.input_features = input_features
        self.output_features = output_features
        self.normalize = normalize
        
        # Extract features
        self.X = data[input_features].values.astype(np.float32)
        self.y = data[output_features].values.astype(np.float32)
        
        # Normalization parameters
        if normalize:
            self.X_mean = self.X.mean(axis=0)
            self.X_std = self.X.std(axis=0) + 1e-8
            self.y_mean = self.y.mean(axis=0)
            self.y_std = self.y.std(axis=0) + 1e-8
            
            # Normalize
            self.X = (self.X - self.X_mean) / self.X_std
            self.y = (self.y - self.y_mean) / self.y_std
        else:
            self.X_mean = np.zeros(self.X.shape[1])
            self.X_std = np.ones(self.X.shape[1])
            self.y_mean = np.zeros(self.y.shape[1])
            self.y_std = np.ones(self.y.shape[1])
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample at index."""
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
    
    def denormalize_output(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalize output values."""
        return y_norm * self.y_std + self.y_mean


def create_dataloaders(
    data: pd.DataFrame,
    input_features: list,
    output_features: list,
    batch_size: int = 128,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, CFDDataset]:
    """
    Create train/val/test dataloaders.
    
    Args:
        data: Pandas DataFrame with CFD data
        input_features: Input feature names
        output_features: Output feature names
        batch_size: Batch size
        train_split: Training fraction
        val_split: Validation fraction
        test_split: Test fraction
        seed: Random seed
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, full_dataset)
    """
    # Create full dataset
    full_dataset = CFDDataset(data, input_features, output_features)
    
    # Split indices
    n = len(full_dataset)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create subsets
    from torch.utils.data import Subset
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)
    
    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, full_dataset
