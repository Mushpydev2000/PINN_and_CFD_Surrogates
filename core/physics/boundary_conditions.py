"""
Boundary Conditions Module
===========================

This module implements various boundary conditions commonly used in
fluid dynamics problems. Boundary conditions are essential constraints
that specify the behavior of the solution at the domain boundaries.

TYPES OF BOUNDARY CONDITIONS:
=============================

1. DIRICHLET BOUNDARY CONDITION (Essential BC):
   Specifies the value of the solution directly on the boundary.
   Example: u = 1.0 at inlet (constant velocity)
   
   Physical Meaning: We know exactly what happens at this boundary.
   Common in: Inlet boundaries, wall boundaries (no-slip condition)

2. NEUMANN BOUNDARY CONDITION (Natural BC):
   Specifies the derivative (gradient) of the solution.
   Example: ∂u/∂n = 0 at outlet (zero gradient)
   
   Physical Meaning: We know the rate of change, not the absolute value.
   Common in: Outlet boundaries, symmetry planes

3. PERIODIC BOUNDARY CONDITION:
   Solution and derivatives repeat periodically.
   Example: u(x, y) = u(x+L, y) for domain of length L
   
   Physical Meaning: Flow pattern repeats in that direction.
   Common in: Channel flows with periodic patterns

4. MIXED BOUNDARY CONDITION:
   Combination of Dirichlet and Neumann conditions.
   Example: a*u + b*∂u/∂n = c
   
   Common in: Robin boundary conditions

WHY BOUNDARY CONDITIONS MATTER:
================================

- Mathematical: PDE requires boundary conditions for well-posedness
- Computational: BCs guide the neural network to physical reality
- Physical: BCs represent how the domain interacts with surroundings

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- BoundaryCondition: Encapsulates related functionality for modularity.
- DirichletBC: Encapsulates related functionality for modularity.
- NeumannBC: Encapsulates related functionality for modularity.
- PeriodicBC: Encapsulates related functionality for modularity.
- BoundaryConditionManager: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
import torch
from torch import Tensor
import numpy as np


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions.
    
    Defines the interface that all boundary condition implementations
    must follow. This allows flexible specification of different BCs
    while maintaining a consistent API.
    """
    
    @abstractmethod
    def apply(
        self,
        u: Tensor,
        v: Tensor,
        p: Tensor,
        x: Tensor,
        y: Tensor,
        **kwargs: Any
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply boundary condition and return residuals.
        
        Args:
            u: X-velocity field [batch_size]
            v: Y-velocity field [batch_size]
            p: Pressure field [batch_size]
            x: X-coordinates [batch_size]
            y: Y-coordinates [batch_size]
            **kwargs: Additional arguments
        
        Returns:
            Tuple of residuals (u_residual, v_residual, p_residual)
            Residual = 0 when BC is satisfied
        """
        pass
    
    @abstractmethod
    def is_on_boundary(self, x: Tensor, y: Tensor, tol: float = 1e-6) -> Tensor:
        """
        Check if points (x, y) are on this boundary.
        
        Args:
            x: X-coordinates [batch_size]
            y: Y-coordinates [batch_size]
            tol: Tolerance for boundary detection
        
        Returns:
            Boolean tensor [batch_size] indicating points on boundary
        """
        pass


class DirichletBC(BoundaryCondition):
    """
    Dirichlet Boundary Condition: Specifies solution values.
    
    Enforces: u = u_value, v = v_value, p = p_value on boundary
    
    Physical Example:
    - Inlet: u=1.0, v=0.0 (flow entering at 1 m/s horizontally)
    - No-slip wall: u=0.0, v=0.0 (stationary boundary)
    
    Implementation:
    Residual = (u_predicted - u_target)² for MSE loss
    
    Attributes:
        boundary_location (str): 'left', 'right', 'top', 'bottom'
        u_value (float): Prescribed u-velocity
        v_value (float): Prescribed v-velocity
        p_value (float): Prescribed pressure (optional)
    """
    
    def __init__(
        self,
        boundary_location: str,
        u_value: float = 0.0,
        v_value: float = 0.0,
        p_value: float = 0.0,
        domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> None:
        """
        Initialize Dirichlet boundary condition.
        
        Args:
            boundary_location: 'left', 'right', 'top', 'bottom', or custom
            u_value: Prescribed u-velocity on boundary
            v_value: Prescribed v-velocity on boundary
            p_value: Prescribed pressure on boundary (optional)
            domain_bounds: Dict with 'x' and 'y' keys containing (min, max) tuples
        """
        self.boundary_location = boundary_location
        self.u_value = u_value
        self.v_value = v_value
        self.p_value = p_value
        self.domain_bounds = domain_bounds or {'x': (-1.0, 1.0), 'y': (-1.0, 1.0)}
    
    def is_on_boundary(
        self,
        x: Tensor,
        y: Tensor,
        tol: float = 1e-6
    ) -> Tensor:
        """
        Identify points on the specified boundary.
        
        Args:
            x: X-coordinates [batch_size]
            y: Y-coordinates [batch_size]
            tol: Tolerance for boundary detection
        
        Returns:
            Boolean tensor indicating points on boundary
        """
        x_min, x_max = self.domain_bounds['x']
        y_min, y_max = self.domain_bounds['y']
        
        if self.boundary_location == 'left':
            return torch.abs(x - x_min) < tol
        elif self.boundary_location == 'right':
            return torch.abs(x - x_max) < tol
        elif self.boundary_location == 'bottom':
            return torch.abs(y - y_min) < tol
        elif self.boundary_location == 'top':
            return torch.abs(y - y_max) < tol
        else:
            return torch.zeros_like(x, dtype=torch.bool)
    
    def apply(
        self,
        u: Tensor,
        v: Tensor,
        p: Tensor,
        x: Tensor,
        y: Tensor,
        **kwargs: Any
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply Dirichlet BC: compute residuals as (predicted - prescribed).
        
        Args:
            u: Predicted u-velocity
            v: Predicted v-velocity
            p: Predicted pressure
            x: X-coordinates
            y: Y-coordinates
        
        Returns:
            Residuals (u_res, v_res, p_res) where residual = predicted - prescribed
        """
        # Compute residuals
        u_residual = u - self.u_value
        v_residual = v - self.v_value
        p_residual = p - self.p_value
        
        return u_residual, v_residual, p_residual


class NeumannBC(BoundaryCondition):
    """
    Neumann Boundary Condition: Specifies gradient (derivative) values.
    
    Enforces: ∂u/∂n = du_dn_value, ∂v/∂n = dv_dn_value on boundary
    
    Physical Example:
    - Outlet: ∂u/∂x = 0 (zero gradient, flow exits freely)
    - Symmetry: ∂u/∂n = 0 (no flow across symmetry plane)
    
    Implementation:
    Residual = (∂u/∂n_predicted - ∂u/∂n_target)² for MSE loss
    
    Note: Computing Neumann BC requires gradient information,
    which is more computationally expensive than Dirichlet.
    
    Attributes:
        boundary_location (str): 'left', 'right', 'top', 'bottom'
        du_dn_value (float): Prescribed ∂u/∂n value
        dv_dn_value (float): Prescribed ∂v/∂n value
    """
    
    def __init__(
        self,
        boundary_location: str,
        du_dn_value: float = 0.0,
        dv_dn_value: float = 0.0,
        domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> None:
        """
        Initialize Neumann boundary condition.
        
        Args:
            boundary_location: 'left', 'right', 'top', 'bottom'
            du_dn_value: Prescribed ∂u/∂n on boundary
            dv_dn_value: Prescribed ∂v/∂n on boundary
            domain_bounds: Domain boundaries
        """
        self.boundary_location = boundary_location
        self.du_dn_value = du_dn_value
        self.dv_dn_value = dv_dn_value
        self.domain_bounds = domain_bounds or {'x': (-1.0, 1.0), 'y': (-1.0, 1.0)}
    
    def is_on_boundary(
        self,
        x: Tensor,
        y: Tensor,
        tol: float = 1e-6
    ) -> Tensor:
        """Identify points on the specified boundary."""
        x_min, x_max = self.domain_bounds['x']
        y_min, y_max = self.domain_bounds['y']
        
        if self.boundary_location == 'left':
            return torch.abs(x - x_min) < tol
        elif self.boundary_location == 'right':
            return torch.abs(x - x_max) < tol
        elif self.boundary_location == 'bottom':
            return torch.abs(y - y_min) < tol
        elif self.boundary_location == 'top':
            return torch.abs(y - y_max) < tol
        else:
            return torch.zeros_like(x, dtype=torch.bool)
    
    def apply(
        self,
        u: Tensor,
        v: Tensor,
        p: Tensor,
        x: Tensor,
        y: Tensor,
        du_dn: Optional[Tensor] = None,
        dv_dn: Optional[Tensor] = None,
        **kwargs: Any
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply Neumann BC: compute residuals as (∂predicted/∂n - ∂prescribed/∂n).
        
        Args:
            u: Predicted u-velocity
            v: Predicted v-velocity
            p: Predicted pressure
            x: X-coordinates
            y: Y-coordinates
            du_dn: Computed ∂u/∂n (normal derivative)
            dv_dn: Computed ∂v/∂n (normal derivative)
        
        Returns:
            Residuals based on normal derivatives
        """
        # If gradients not provided, return zero residuals
        if du_dn is None or dv_dn is None:
            return (
                torch.zeros_like(u),
                torch.zeros_like(v),
                torch.zeros_like(p)
            )
        
        # Compute residuals
        u_residual = du_dn - self.du_dn_value
        v_residual = dv_dn - self.dv_dn_value
        p_residual = torch.zeros_like(p)
        
        return u_residual, v_residual, p_residual


class PeriodicBC(BoundaryCondition):
    """
    Periodic Boundary Condition: Solution repeats across boundary.
    
    Enforces: u(x, y) = u(x+L, y), v(x, y) = v(x+L, y) for periodic domain
    
    Physical Example:
    - Channel flow with periodic inlet/outlet
    - Fully developed pipe flow
    - Pattern repeats endlessly in one direction
    
    Implementation:
    Residual = (value_at_x - value_at_x+L)²
    
    Attributes:
        direction (str): 'x' or 'y' (which direction is periodic)
        period (float): Length of periodic domain
    """
    
    def __init__(
        self,
        direction: str = 'x',
        period: float = 2.0,
        domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> None:
        """
        Initialize periodic boundary condition.
        
        Args:
            direction: 'x' or 'y' for periodic direction
            period: Length of periodic domain
            domain_bounds: Domain boundaries
        """
        assert direction in ['x', 'y'], "Direction must be 'x' or 'y'"
        self.direction = direction
        self.period = period
        self.domain_bounds = domain_bounds or {'x': (-1.0, 1.0), 'y': (-1.0, 1.0)}
    
    def is_on_boundary(
        self,
        x: Tensor,
        y: Tensor,
        tol: float = 1e-6
    ) -> Tensor:
        """Identify points on periodic boundaries."""
        if self.direction == 'x':
            x_min = self.domain_bounds['x'][0]
            return torch.abs(x - x_min) < tol
        else:
            y_min = self.domain_bounds['y'][0]
            return torch.abs(y - y_min) < tol
    
    def apply(
        self,
        u: Tensor,
        v: Tensor,
        p: Tensor,
        x: Tensor,
        y: Tensor,
        u_shifted: Optional[Tensor] = None,
        v_shifted: Optional[Tensor] = None,
        **kwargs: Any
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply periodic BC: enforce solution periodicity.
        
        Args:
            u: Velocity at boundary 1
            v: Velocity at boundary 1
            p: Pressure at boundary 1
            x: X-coordinates
            y: Y-coordinates
            u_shifted: Velocity at boundary 2 (shifted by period)
            v_shifted: Velocity at boundary 2 (shifted by period)
        
        Returns:
            Residuals enforcing periodicity
        """
        if u_shifted is None or v_shifted is None:
            return (
                torch.zeros_like(u),
                torch.zeros_like(v),
                torch.zeros_like(p)
            )
        
        # Residuals enforce u(x) = u(x + period)
        u_residual = u - u_shifted
        v_residual = v - v_shifted
        p_residual = torch.zeros_like(p)
        
        return u_residual, v_residual, p_residual


class BoundaryConditionManager:
    """
    Manages multiple boundary conditions.
    
    Collects all BCs for a domain and applies them during training.
    This class provides a convenient interface for handling complex
    problems with multiple boundary condition types.
    
    Example:
        bc_mgr = BoundaryConditionManager()
        bc_mgr.add_bc(DirichletBC('left', u_value=1.0))  # Inlet
        bc_mgr.add_bc(DirichletBC('right'))  # Wall
        
        # During training:
        residuals = bc_mgr.compute_residuals(u, v, p, x, y)
    
    Attributes:
        boundary_conditions (List[BoundaryCondition]): List of all BCs
    """
    
    def __init__(self) -> None:
        """Initialize boundary condition manager."""
        self.boundary_conditions: List[BoundaryCondition] = []
    
    def add_bc(self, bc: BoundaryCondition) -> None:
        """
        Add a boundary condition to the manager.
        
        Args:
            bc: BoundaryCondition object
        """
        self.boundary_conditions.append(bc)
    
    def compute_residuals(
        self,
        u: Tensor,
        v: Tensor,
        p: Tensor,
        x: Tensor,
        y: Tensor,
        **kwargs: Any
    ) -> Tensor:
        """
        Compute all boundary condition residuals.
        
        Aggregates residuals from all boundary conditions into a single
        tensor for loss computation.
        
        Args:
            u: Velocity field
            v: Velocity field
            p: Pressure field
            x: X-coordinates
            y: Y-coordinates
        
        Returns:
            Concatenated residuals from all boundary conditions
        """
        all_residuals = []
        
        for bc in self.boundary_conditions:
            # Get points on this boundary
            on_boundary = bc.is_on_boundary(x, y)
            
            if on_boundary.sum() > 0:
                # Extract boundary points
                u_bc = u[on_boundary]
                v_bc = v[on_boundary]
                p_bc = p[on_boundary]
                x_bc = x[on_boundary]
                y_bc = y[on_boundary]
                
                # Apply BC and get residuals
                u_res, v_res, p_res = bc.apply(
                    u_bc, v_bc, p_bc, x_bc, y_bc, **kwargs
                )
                
                # Combine residuals
                combined_res = torch.cat([u_res, v_res, p_res], dim=0)
                all_residuals.append(combined_res)
        
        if all_residuals:
            return torch.cat(all_residuals, dim=0)
        else:
            return torch.tensor([], device=u.device)
    
    def __len__(self) -> int:
        """Return number of boundary conditions."""
        return len(self.boundary_conditions)
    
    def __repr__(self) -> str:
        """String representation of boundary condition manager."""
        return (
            f"BoundaryConditionManager with {len(self.boundary_conditions)} "
            f"boundary conditions"
        )
