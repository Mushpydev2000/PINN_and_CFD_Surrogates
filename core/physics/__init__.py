"""Physics module for Navier-Stokes equations and fluid dynamics."""

from .navier_stokes import NavierStokesEquations2D
from .boundary_conditions import (
    BoundaryCondition,
    DirichletBC,
    NeumannBC,
    PeriodicBC,
    BoundaryConditionManager,
)
from .pde_losses import PINNLoss, AdaptiveWeightLoss
from .fluid_properties import (
    FluidProperties,
    FluidDatabase,
    compute_flow_regimes,
    WATER,
    AIR,
    OIL,
    GLYCEROL,
    MERCURY,
)

__all__ = [
    'NavierStokesEquations2D',
    'BoundaryCondition',
    'DirichletBC',
    'NeumannBC',
    'PeriodicBC',
    'BoundaryConditionManager',
    'PINNLoss',
    'AdaptiveWeightLoss',
    'FluidProperties',
    'FluidDatabase',
    'compute_flow_regimes',
    'WATER',
    'AIR',
    'OIL',
    'GLYCEROL',
    'MERCURY',
]
