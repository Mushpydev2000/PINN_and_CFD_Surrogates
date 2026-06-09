"""Unit tests for physics module."""

import pytest
import torch
import numpy as np
from core.physics import NavierStokesEquations2D, FluidProperties, WATER, AIR


class TestNavierStokesEquations:
    """Test Navier-Stokes equation implementations."""
    
    def test_initialization(self):
        """Test NS equations initialization."""
        ns = NavierStokesEquations2D(rho=1.0, mu=0.01)
        assert ns.rho == 1.0
        assert ns.mu == 0.01
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError):
            NavierStokesEquations2D(rho=-1.0, mu=0.01)
        
        with pytest.raises(ValueError):
            NavierStokesEquations2D(rho=1.0, mu=-0.01)
    
    def test_continuity_residual(self):
        """Test continuity equation residual."""
        ns = NavierStokesEquations2D()
        
        # Create simple test tensors
        du_dx = torch.tensor([0.1, 0.2])
        dv_dy = torch.tensor([-0.1, -0.2])
        
        residual = ns.continuity_residual(du_dx, dv_dy)
        
        # Should be ~zero for zero divergence
        assert residual.shape == du_dx.shape


class TestFluidProperties:
    """Test fluid properties."""
    
    def test_water_properties(self):
        """Test water properties."""
        assert WATER.density == 1000.0
        assert WATER.dynamic_viscosity == 0.001
    
    def test_reynolds_number(self):
        """Test Reynolds number computation."""
        re = WATER.reynolds_number(velocity=1.0, characteristic_length=0.1)
        
        # For water: Re = ρ * U * L / μ = 1000 * 1.0 * 0.1 / 0.001 = 100000
        expected_re = 100000.0
        assert abs(re - expected_re) < 1e-6
    
    def test_prandtl_number(self):
        """Test Prandtl number computation."""
        pr = WATER.prandtl_number()
        assert pr > 0  # Should be positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
