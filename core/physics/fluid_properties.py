"""
Fluid Properties Module
========================

This module defines fluid properties and dimensionless numbers used in
fluid dynamics. These properties are essential for setting up physical
simulations and understanding flow regimes.

KEY DIMENSIONLESS NUMBERS:
===========================

1. REYNOLDS NUMBER (Re):
   Re = ρ * U * L / μ = U * L / ν
   
   Physical Meaning: Ratio of inertial to viscous forces
   - Re << 1: Viscous forces dominate (creeping flow)
   - Re ~ 1: Balanced forces (Stokes flow)
   - Re >> 1: Inertial forces dominate (turbulent)
   
   Example:
   - Swimming bacterium in water: Re ~ 1e-4 (creeping flow)
   - Fish swimming: Re ~ 1e3 (turbulent)
   - Airplane flight: Re ~ 1e7 (highly turbulent)

2. FROUDE NUMBER (Fr):
   Fr = U / √(g * L)
   
   Physical Meaning: Ratio of inertial to gravitational forces
   Used when gravity is important (free surface flows, ships)

3. MACH NUMBER (M):
   M = U / a (where a is speed of sound)
   
   Physical Meaning: Ratio of fluid velocity to sound speed
   - M < 1: Subsonic (compressibility effects small)
   - M > 1: Supersonic (shockwaves form)

4. STROUHAL NUMBER (St):
   St = f * L / U
   
   Physical Meaning: Frequency of oscillations relative to flow
   Used in vortex shedding and periodic flows

Author: Scientific ML Research Group
Version: 1.0.0


ENCAPSULATED CLASSES & ABSTRACTIONS:
- FluidProperties: Encapsulates related functionality for modularity.
- FluidDatabase: Encapsulates related functionality for modularity.

DATA STRUCTURES & RATIONALE:
- PyTorch Tensors: Used extensively for automatic differentiation (autograd) and GPU acceleration.
- NumPy Arrays/Pandas DataFrames: Used for data processing and interfacing with visualization libraries.
- Dictionaries: Used for flexible configuration and mapping (e.g., loss histories, parameters).

DESIGN RATIONALE:
This file is part of the modular architecture separating physics definitions, neural networks,
and training pipelines. This separation of concerns allows for easy substitution of PDE
equations, network architectures, and training strategies without tight coupling.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np


@dataclass
class FluidProperties:
    """
    Container for fluid properties.
    
    Stores common fluid properties used in simulations and provides
    utility methods for computing derived quantities like viscosity
    coefficients and dimensionless numbers.
    
    Attributes:
        name (str): Fluid name (e.g., 'water', 'air', 'oil')
        density (float): Mass density [kg/m³]
        dynamic_viscosity (float): Dynamic viscosity μ [Pa·s = kg/(m·s)]
        kinematic_viscosity (float): Kinematic viscosity ν [m²/s]
        specific_heat (float): Specific heat capacity [J/(kg·K)]
        thermal_conductivity (float): Thermal conductivity [W/(m·K)]
        temperature (float): Reference temperature [K]
    """
    
    name: str
    density: float  # [kg/m³]
    dynamic_viscosity: float  # [Pa·s]
    kinematic_viscosity: Optional[float] = None  # [m²/s]
    specific_heat: float = 1000.0  # [J/(kg·K)]
    thermal_conductivity: float = 0.6  # [W/(m·K)]
    temperature: float = 293.15  # [K] (20°C)
    
    def __post_init__(self) -> None:
        """Post-initialization: compute kinematic viscosity if not provided."""
        if self.kinematic_viscosity is None:
            if self.density <= 0:
                raise ValueError("Density must be positive")
            self.kinematic_viscosity = self.dynamic_viscosity / self.density
    
    def reynolds_number(
        self,
        velocity: float,
        characteristic_length: float
    ) -> float:
        """
        Compute Reynolds number.
        
        Re = ρ * U * L / μ = U * L / ν
        
        Physical Interpretation:
        - Re << 1: Viscous flow (Stokes regime)
        - Re ~ 1: Transition
        - Re >> 1: Inertial flow (possible turbulence)
        
        Args:
            velocity: Characteristic velocity [m/s]
            characteristic_length: Characteristic length [m]
        
        Returns:
            Reynolds number (dimensionless)
        """
        return (self.density * velocity * characteristic_length) / self.dynamic_viscosity
    
    def froude_number(
        self,
        velocity: float,
        characteristic_length: float,
        gravity: float = 9.81
    ) -> float:
        """
        Compute Froude number.
        
        Fr = U / √(g * L)
        
        Physical Interpretation:
        - Fr << 1: Gravity dominates (shallow water, internal waves)
        - Fr ~ 1: Balanced (critical flow)
        - Fr >> 1: Inertia dominates (deep water, rapid flow)
        
        Args:
            velocity: Characteristic velocity [m/s]
            characteristic_length: Characteristic length [m]
            gravity: Gravitational acceleration [m/s²] (default: 9.81)
        
        Returns:
            Froude number (dimensionless)
        """
        if characteristic_length <= 0:
            raise ValueError("Characteristic length must be positive")
        
        return velocity / np.sqrt(gravity * characteristic_length)
    
    def mach_number(
        self,
        velocity: float,
        speed_of_sound: Optional[float] = None
    ) -> float:
        """
        Compute Mach number.
        
        M = U / a
        
        Physical Interpretation:
        - M << 1: Subsonic (compressibility effects negligible)
        - M ~ 1: Transonic (complex flow)
        - M >> 1: Supersonic (shockwaves)
        
        Args:
            velocity: Characteristic velocity [m/s]
            speed_of_sound: Speed of sound [m/s]
                           If None, computed from fluid properties
        
        Returns:
            Mach number (dimensionless)
        """
        if speed_of_sound is None:
            # For air at 20°C: approximately 343 m/s
            # For water: approximately 1481 m/s
            if 'water' in self.name.lower():
                speed_of_sound = 1481.0
            else:  # Default to air
                speed_of_sound = 343.0
        
        return velocity / speed_of_sound
    
    def strouhal_number(
        self,
        frequency: float,
        velocity: float,
        characteristic_length: float
    ) -> float:
        """
        Compute Strouhal number.
        
        St = f * L / U
        
        Physical Interpretation:
        - St << 1: Quasi-steady behavior
        - St ~ 0.1-0.2: Vortex shedding from cylinder
        - St >> 1: Highly periodic oscillations
        
        Args:
            frequency: Oscillation frequency [Hz]
            velocity: Characteristic velocity [m/s]
            characteristic_length: Characteristic length [m]
        
        Returns:
            Strouhal number (dimensionless)
        """
        if velocity <= 0:
            raise ValueError("Velocity must be positive")
        
        return (frequency * characteristic_length) / velocity
    
    def prandtl_number(self) -> float:
        """
        Compute Prandtl number.
        
        Pr = ν / α = (μ * cp) / k
        
        Physical Interpretation:
        - Pr << 1: Heat diffuses faster than momentum (liquid metals)
        - Pr ~ 1: Heat and momentum diffuse equally
        - Pr >> 1: Momentum diffuses faster than heat (oils, polymers)
        
        Returns:
            Prandtl number (dimensionless)
        """
        # Thermal diffusivity α = k / (ρ * cp)
        thermal_diffusivity = self.thermal_conductivity / (
            self.density * self.specific_heat
        )
        
        return self.kinematic_viscosity / thermal_diffusivity
    
    def __repr__(self) -> str:
        """String representation of fluid properties."""
        return (
            f"FluidProperties(name='{self.name}', "
            f"ρ={self.density:.1f} kg/m³, "
            f"μ={self.dynamic_viscosity:.4e} Pa·s, "
            f"ν={self.kinematic_viscosity:.4e} m²/s)"
        )


# Pre-defined fluid properties for common fluids at standard conditions

WATER = FluidProperties(
    name="Water",
    density=1000.0,  # [kg/m³]
    dynamic_viscosity=0.001,  # [Pa·s] at 20°C
    specific_heat=4186.0,  # [J/(kg·K)]
    thermal_conductivity=0.6,  # [W/(m·K)]
    temperature=293.15  # [K] (20°C)
)

AIR = FluidProperties(
    name="Air",
    density=1.225,  # [kg/m³] at 15°C, 1 atm
    dynamic_viscosity=1.81e-5,  # [Pa·s] at 15°C
    specific_heat=1005.0,  # [J/(kg·K)]
    thermal_conductivity=0.0257,  # [W/(m·K)]
    temperature=288.15  # [K] (15°C)
)

OIL = FluidProperties(
    name="Oil",
    density=850.0,  # [kg/m³]
    dynamic_viscosity=0.1,  # [Pa·s] (SAE 30 oil)
    specific_heat=1900.0,  # [J/(kg·K)]
    thermal_conductivity=0.145,  # [W/(m·K)],
    temperature=293.15  # [K]
)

GLYCEROL = FluidProperties(
    name="Glycerol",
    density=1260.0,  # [kg/m³]
    dynamic_viscosity=1.0,  # [Pa·s] (highly viscous)
    specific_heat=2430.0,  # [J/(kg·K)]
    thermal_conductivity=0.285,  # [W/(m·K)]
    temperature=293.15  # [K]
)

MERCURY = FluidProperties(
    name="Mercury",
    density=13600.0,  # [kg/m³]
    dynamic_viscosity=0.0015,  # [Pa·s]
    specific_heat=140.0,  # [J/(kg·K)]
    thermal_conductivity=8.3,  # [W/(m·K)]
    temperature=293.15  # [K]
)


class FluidDatabase:
    """
    Database of predefined fluid properties.
    
    Provides convenient access to properties of common fluids,
    eliminating need to specify properties manually.
    
    Example:
        db = FluidDatabase()
        water = db.get_fluid('water')
        re = water.reynolds_number(velocity=1.0, characteristic_length=0.1)
    """
    
    def __init__(self) -> None:
        """Initialize fluid database with common fluids."""
        self.fluids: Dict[str, FluidProperties] = {
            'water': WATER,
            'air': AIR,
            'oil': OIL,
            'glycerol': GLYCEROL,
            'mercury': MERCURY
        }
    
    def get_fluid(self, name: str) -> FluidProperties:
        """
        Get fluid properties by name.
        
        Args:
            name: Fluid name (case-insensitive)
                 Available: 'water', 'air', 'oil', 'glycerol', 'mercury'
        
        Returns:
            FluidProperties object
        
        Raises:
            KeyError: If fluid not found in database
        """
        name_lower = name.lower()
        if name_lower not in self.fluids:
            available = ', '.join(self.fluids.keys())
            raise KeyError(
                f"Fluid '{name}' not found. Available fluids: {available}"
            )
        return self.fluids[name_lower]
    
    def add_fluid(self, fluid: FluidProperties) -> None:
        """
        Add custom fluid to database.
        
        Args:
            fluid: FluidProperties object
        """
        self.fluids[fluid.name.lower()] = fluid
    
    def list_fluids(self) -> list:
        """List all available fluids."""
        return list(self.fluids.keys())
    
    def __repr__(self) -> str:
        """String representation of database."""
        return (
            f"FluidDatabase with {len(self.fluids)} fluids: "
            f"{', '.join(self.fluids.keys())}"
        )


def compute_flow_regimes(
    reynolds_number: float
) -> Dict[str, str]:
    """
    Classify flow regime based on Reynolds number.
    
    Provides physical interpretation of Reynolds number values.
    
    Args:
        reynolds_number: Reynolds number value
    
    Returns:
        Dictionary with flow regime information
    """
    if reynolds_number < 1:
        regime = "Creeping Flow (Stokes Flow)"
        description = "Viscous forces dominate, inertia negligible"
    elif 1 <= reynolds_number < 1000:
        regime = "Laminar Flow"
        description = "Ordered, smooth fluid motion"
    elif 1000 <= reynolds_number < 100000:
        regime = "Transitional Flow"
        description = "Mix of laminar and turbulent characteristics"
    else:
        regime = "Turbulent Flow"
        description = "Chaotic, high-energy fluid motion"
    
    return {
        'regime': regime,
        'description': description,
        'reynolds_number': reynolds_number
    }
