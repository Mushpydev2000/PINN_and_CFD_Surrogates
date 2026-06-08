"""
Physics Module: Navier-Stokes Equations for 2D Incompressible Fluid Flow
==========================================================================

This module implements the fundamental equations governing 2D incompressible 
Newtonian fluid flow. It provides the mathematical framework for physics-informed 
neural networks to solve fluid dynamics problems.

MATHEMATICAL BACKGROUND:
=======================

For a 2D incompressible Newtonian fluid, the governing equations are:

1. CONTINUITY EQUATION (Mass Conservation):
   ∂u/∂x + ∂v/∂y = 0
   
   Physical Meaning: The sum of velocity gradients must equal zero to ensure
   mass is conserved in an incompressible fluid. No fluid is created or destroyed.
   
   Expected Output: Small residual value (ideally ~0) when continuity is satisfied

2. MOMENTUM EQUATIONS (Newton's Second Law: F = ma):
   
   X-MOMENTUM:
   ρ(∂u/∂t + u∂u/∂x + v∂u/∂y) = -∂p/∂x + μ(∂²u/∂x² + ∂²u/∂y²) + f_x
   
   Y-MOMENTUM:
   ρ(∂v/∂t + u∂v/∂x + v∂v/∂y) = -∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y²) + f_y
   
   Physical Meaning: 
   - Left side: Acceleration (temporal + convective terms)
   - Right side: Forces acting on the fluid
     * Pressure gradient: -∇p (drives flow)
     * Viscous forces: μ∇²u (resists motion)
     * Body forces: f (external forces like gravity)
   
   Expected Output: Small residual (ideally ~0) when momentum is conserved

WHY THESE EQUATIONS:
====================

1. Continuity: Without it, mass conservation is violated (physically impossible)
2. Momentum X: Governs horizontal velocity evolution
3. Momentum Y: Governs vertical velocity evolution
4. Together: Complete system to determine u(x,y,t), v(x,y,t), p(x,y,t)

KEY PARAMETERS:
===============

- ρ (rho): Fluid density [kg/m³]. Ratio affects inertial vs viscous forces
- μ (mu): Dynamic viscosity [Pa·s]. Determines fluid resistance to flow
- Re = ρUD/μ: Reynolds number. Ratio of inertial to viscous forces
  * Re << 1: Creeping flow (viscosity dominates)
  * Re ~ 1-1000: Laminar flow with complex patterns
  * Re >> 1000: Turbulent flow (very complex)

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- NavierStokesEquations2D: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from torch import Tensor


class NavierStokesEquations2D:
    """
    2D Incompressible Navier-Stokes Equations Solver.
    
    This class encapsulates the mathematical formulation of 2D incompressible
    Navier-Stokes equations and provides methods to compute residuals for
    physics-informed learning.
    
    The governing equations are:
    - Continuity: ∂u/∂x + ∂v/∂y = 0
    - X-Momentum: ρ(∂u/∂t + u∂u/∂x + v∂u/∂y) = -∂p/∂x + μ∇²u
    - Y-Momentum: ρ(∂v/∂t + u∂v/∂x + v∂v/∂y) = -∂p/∂y + μ∇²v
    
    Attributes:
        rho (float): Fluid density [kg/m³]
        mu (float): Dynamic viscosity [Pa·s]
        reynolds_number (float): Re = ρ * U_ref * L_ref / μ
    """
    
    def __init__(
        self,
        rho: float = 1.0,
        mu: float = 0.01,
        reynolds_number: Optional[float] = None
    ) -> None:
        """
        Initialize Navier-Stokes equations.
        
        Args:
            rho: Fluid density [kg/m³]. Default: 1.0 (water-like at normalized scale)
            mu: Dynamic viscosity [Pa·s]. Default: 0.01
            reynolds_number: Reynolds number. If provided, mu is computed as 
                           rho * U_ref * L_ref / Re
        
        Raises:
            ValueError: If negative density or viscosity provided
        """
        if rho <= 0 or mu <= 0:
            raise ValueError("Density and viscosity must be positive")
        
        self.rho = rho
        self.mu = mu
        self.reynolds_number = reynolds_number
    
    @staticmethod
    def compute_derivatives(
        u: Tensor,
        v: Tensor,
        p: Tensor,
        coords: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute first-order spatial derivatives using automatic differentiation.

        Gradients are taken with respect to the full coordinate tensor so the
        autograd graph remains connected through the neural network forward pass.

        Args:
            u: X-velocity field [batch_size, 1]
            v: Y-velocity field [batch_size, 1]
            p: Pressure field [batch_size, 1]
            coords: Collocation coordinates [batch_size, 2] with (x, y)

        Returns:
            Tuple of du_dx, du_dy, dv_dx, dv_dy, dp_dx, dp_dy
        """
        grad_u = torch.autograd.grad(
            u.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        grad_v = torch.autograd.grad(
            v.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        grad_p = torch.autograd.grad(
            p.sum(), coords, create_graph=True, retain_graph=True
        )[0]

        du_dx = grad_u[:, 0:1]
        du_dy = grad_u[:, 1:2]
        dv_dx = grad_v[:, 0:1]
        dv_dy = grad_v[:, 1:2]
        dp_dx = grad_p[:, 0:1]
        dp_dy = grad_p[:, 1:2]

        return du_dx, du_dy, dv_dx, dv_dy, dp_dx, dp_dy

    @staticmethod
    def compute_second_derivatives(
        u: Tensor,
        v: Tensor,
        coords: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute second-order spatial derivatives (Laplacian terms).

        Args:
            u: X-velocity field [batch_size, 1]
            v: Y-velocity field [batch_size, 1]
            coords: Collocation coordinates [batch_size, 2] with (x, y)

        Returns:
            Tuple of d2u_dx2, d2u_dy2, d2v_dx2, d2v_dy2
        """
        grad_u = torch.autograd.grad(
            u.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        grad_v = torch.autograd.grad(
            v.sum(), coords, create_graph=True, retain_graph=True
        )[0]

        du_dx = grad_u[:, 0:1]
        du_dy = grad_u[:, 1:2]
        dv_dx = grad_v[:, 0:1]
        dv_dy = grad_v[:, 1:2]

        grad_du_dx = torch.autograd.grad(
            du_dx.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        grad_du_dy = torch.autograd.grad(
            du_dy.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        grad_dv_dx = torch.autograd.grad(
            dv_dx.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        grad_dv_dy = torch.autograd.grad(
            dv_dy.sum(), coords, create_graph=True, retain_graph=True
        )[0]

        d2u_dx2 = grad_du_dx[:, 0:1]
        d2u_dy2 = grad_du_dy[:, 1:2]
        d2v_dx2 = grad_dv_dx[:, 0:1]
        d2v_dy2 = grad_dv_dy[:, 1:2]

        return d2u_dx2, d2u_dy2, d2v_dx2, d2v_dy2
    
    def continuity_residual(
        self,
        du_dx: Tensor,
        dv_dy: Tensor
    ) -> Tensor:
        """
        Compute continuity equation residual: ∂u/∂x + ∂v/∂y = 0
        
        Physical Interpretation:
        - Residual = 0: Mass is conserved (incompressible condition satisfied)
        - Residual ≠ 0: Artificial mass creation/destruction (physically invalid)
        
        The continuity equation ensures that for an incompressible fluid,
        the divergence of velocity is zero. In a fixed control volume,
        what flows in must flow out.
        
        Args:
            du_dx: ∂u/∂x tensor [batch_size]
            dv_dy: ∂v/∂y tensor [batch_size]
        
        Returns:
            Residual: ∂u/∂x + ∂v/∂y [batch_size]
        """
        # Continuity equation: ∂u/∂x + ∂v/∂y = 0
        residual = du_dx + dv_dy
        return residual
    
    def x_momentum_residual(
        self,
        du_dx: Tensor,
        du_dy: Tensor,
        d2u_dx2: Tensor,
        d2u_dy2: Tensor,
        u: Tensor,
        v: Tensor,
        dp_dx: Tensor,
        du_dt: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute X-momentum equation residual.
        
        ρ(∂u/∂t + u∂u/∂x + v∂u/∂y) = -∂p/∂x + μ(∂²u/∂x² + ∂²u/∂y²)
        
        Rearranged for residual = 0:
        ρ(∂u/∂t + u∂u/∂x + v∂u/∂y) + ∂p/∂x - μ∇²u = 0
        
        Physical Interpretation:
        - Left side (acceleration): How fast the fluid particle is being accelerated
        - Right side (forces):
          * Pressure gradient: Pushes fluid from high to low pressure
          * Viscous term: Resists motion (internal friction)
        
        Args:
            du_dx: ∂u/∂x
            du_dy: ∂u/∂y
            d2u_dx2: ∂²u/∂x²
            d2u_dy2: ∂²u/∂y²
            u: X-velocity
            v: Y-velocity
            dp_dx: ∂p/∂x
            du_dt: ∂u/∂t (if time-dependent, otherwise None for steady-state)
        
        Returns:
            Residual: ρ(∂u/∂t + u∂u/∂x + v∂u/∂y) + ∂p/∂x - μ∇²u
        """
        # Convective acceleration: u∂u/∂x + v∂u/∂y
        convective_accel = u * du_dx + v * du_dy
        
        # Temporal acceleration (if time-dependent)
        if du_dt is not None:
            temporal_accel = du_dt
        else:
            temporal_accel = torch.zeros_like(u)
        
        # Total acceleration (with density)
        acceleration = self.rho * (temporal_accel + convective_accel)
        
        # Viscous term: μ∇²u = μ(∂²u/∂x² + ∂²u/∂y²)
        laplacian_u = d2u_dx2 + d2u_dy2
        viscous_force = self.mu * laplacian_u
        
        # X-Momentum residual: acceleration + ∇p - viscous_force = 0
        residual = acceleration + dp_dx - viscous_force
        
        return residual
    
    def y_momentum_residual(
        self,
        dv_dx: Tensor,
        dv_dy: Tensor,
        d2v_dx2: Tensor,
        d2v_dy2: Tensor,
        u: Tensor,
        v: Tensor,
        dp_dy: Tensor,
        dv_dt: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute Y-momentum equation residual.
        
        ρ(∂v/∂t + u∂v/∂x + v∂v/∂y) = -∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y²)
        
        Rearranged for residual = 0:
        ρ(∂v/∂t + u∂v/∂x + v∂v/∂y) + ∂p/∂y - μ∇²v = 0
        
        Physical Interpretation: Same as X-momentum but for vertical direction.
        
        Args:
            dv_dx: ∂v/∂x
            dv_dy: ∂v/∂y
            d2v_dx2: ∂²v/∂x²
            d2v_dy2: ∂²v/∂y²
            u: X-velocity
            v: Y-velocity
            dp_dy: ∂p/∂y
            dv_dt: ∂v/∂t (if time-dependent, otherwise None for steady-state)
        
        Returns:
            Residual: ρ(∂v/∂t + u∂v/∂x + v∂v/∂y) + ∂p/∂y - μ∇²v
        """
        # Convective acceleration: u∂v/∂x + v∂v/∂y
        convective_accel = u * dv_dx + v * dv_dy
        
        # Temporal acceleration (if time-dependent)
        if dv_dt is not None:
            temporal_accel = dv_dt
        else:
            temporal_accel = torch.zeros_like(v)
        
        # Total acceleration (with density)
        acceleration = self.rho * (temporal_accel + convective_accel)
        
        # Viscous term: μ∇²v = μ(∂²v/∂x² + ∂²v/∂y²)
        laplacian_v = d2v_dx2 + d2v_dy2
        viscous_force = self.mu * laplacian_v
        
        # Y-Momentum residual: acceleration + ∇p - viscous_force = 0
        residual = acceleration + dp_dy - viscous_force
        
        return residual
    
    def compute_all_residuals(
        self,
        u: Tensor,
        v: Tensor,
        p: Tensor,
        coords: Tensor,
        du_dt: Optional[Tensor] = None,
        dv_dt: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute all three PDE residuals in one call.
        
        This is the main interface for computing physics residuals used in
        the PINN loss function during training.
        
        Args:
            u: X-velocity predictions from neural network
            v: Y-velocity predictions from neural network
            p: Pressure predictions from neural network
            coords: Collocation coordinates [batch_size, 2] with (x, y)
            du_dt: ∂u/∂t if time-dependent
            dv_dt: ∂v/∂t if time-dependent
        
        Returns:
            Tuple of:
            - continuity_res: Continuity equation residual
            - x_momentum_res: X-momentum equation residual
            - y_momentum_res: Y-momentum equation residual
        """
        if not coords.requires_grad:
            coords = coords.requires_grad_(True)

        # Compute first-order derivatives
        du_dx, du_dy, dv_dx, dv_dy, dp_dx, dp_dy = self.compute_derivatives(
            u, v, p, coords
        )

        # Compute second-order derivatives
        d2u_dx2, d2u_dy2, d2v_dx2, d2v_dy2 = self.compute_second_derivatives(
            u, v, coords
        )
        
        # Compute residuals
        continuity_res = self.continuity_residual(du_dx, dv_dy)
        
        x_momentum_res = self.x_momentum_residual(
            du_dx, du_dy, d2u_dx2, d2u_dy2, u, v, dp_dx, du_dt
        )
        
        y_momentum_res = self.y_momentum_residual(
            dv_dx, dv_dy, d2v_dx2, d2v_dy2, u, v, dp_dy, dv_dt
        )
        
        return continuity_res, x_momentum_res, y_momentum_res
