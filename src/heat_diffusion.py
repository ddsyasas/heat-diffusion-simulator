"""
Heat Diffusion Simulator

This module implements a numerical simulation of heat diffusion in a 1D material
using the Forward-Time Central-Space (FTCS) finite difference method.

The heat equation is: ∂T/∂t = α * ∂²T/∂x²
where:
- T is temperature
- t is time
- x is position
- α is thermal diffusivity
"""

import numpy as np
from typing import Tuple
import warnings


class HeatDiffusionSimulator:
    """
    A class to simulate heat diffusion in a 1D material using the FTCS method.
    
    This is the basic implementation for Day 2 - core physics only.
    """
    
    def __init__(self, alpha: float, L: float, nx: int, t_end: float):
        """
        Initialize the heat diffusion simulator.
        
        Parameters:
        -----------
        alpha : float
            Thermal diffusivity of the material (m²/s)
        L : float
            Total length of the material (m)
        nx : int
            Number of grid points along the rod
        t_end : float
            Total simulation time (s)
        """
        self.alpha = alpha
        self.L = L
        self.nx = nx
        self.t_end = t_end
        
        # Create spatial grid
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)
        
        # Initialize temperature array
        self.T = np.zeros(nx)
        
        # Calculate maximum stable time step (CFL condition)
        # For FTCS: dt <= 0.5 * dx² / α
        self.dt_max = 0.5 * self.dx**2 / alpha
        self.dt = self.dt_max  # Use maximum stable time step
        
        # Calculate number of time steps
        self.nt = int(np.ceil(t_end / self.dt))
        self.dt = t_end / self.nt  # Adjust dt to exactly reach t_end
        
        # Initialize time array
        self.t = np.linspace(0, t_end, self.nt + 1)
        
        # Storage for temperature history
        self.T_history = np.zeros((self.nt + 1, nx))
        
        print(f"HeatDiffusionSimulator initialized:")
        print(f"  Grid points: {nx}")
        print(f"  Spatial step: {self.dx:.6f} m")
        print(f"  Time steps: {self.nt}")
        print(f"  Time step: {self.dt:.6f} s")
        print(f"  Max stable step: {self.dt_max:.6f} s")
    
    def set_initial_condition(self, T_initial: np.ndarray) -> None:
        """
        Set the initial temperature distribution.
        
        Parameters:
        -----------
        T_initial : np.ndarray
            Initial temperature values at each grid point
        """
        if len(T_initial) != self.nx:
            raise ValueError(f"Initial condition length {len(T_initial)} must match grid size {self.nx}")
        
        self.T = T_initial.copy()
        self.T_history[0] = self.T.copy()
        print(f"Initial condition set with temperature range: [{self.T.min():.3f}, {self.T.max():.3f}] K")
    
    def apply_boundary_conditions(self, bc_type: str, 
                                left_temp: float = 0.0, 
                                right_temp: float = 0.0) -> None:
        """
        Apply boundary conditions to the temperature array.
        
        Parameters:
        -----------
        bc_type : str
            Type of boundary condition ('dirichlet' or 'neumann')
        left_temp : float, optional
            Left boundary temperature for Dirichlet BC (default: 0.0)
        right_temp : float, optional
            Right boundary temperature for Dirichlet BC (default: 0.0)
        """
        if bc_type.lower() == 'dirichlet':
            self.T[0] = left_temp
            self.T[-1] = right_temp
        elif bc_type.lower() == 'neumann':
            # Insulated boundaries: ∂T/∂x = 0
            # This means T[0] = T[1] and T[-1] = T[-2]
            self.T[0] = self.T[1]
            self.T[-1] = self.T[-2]
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")
    
    def ftcs_step(self) -> None:
        """
        Perform one time step using the FTCS method.
        
        The FTCS scheme is:
        T^(n+1)_i = T^n_i + (α * dt / dx²) * (T^n_(i+1) - 2*T^n_i + T^n_(i-1))
        """
        # Create a copy for the update
        T_new = self.T.copy()
        
        # Apply FTCS scheme to interior points
        for i in range(1, self.nx - 1):
            T_new[i] = (self.T[i] + 
                       (self.alpha * self.dt / self.dx**2) * 
                       (self.T[i+1] - 2*self.T[i] + self.T[i-1]))
        
        # Update temperature array
        self.T = T_new
    
    def simulate(self, T_initial: np.ndarray, bc_type: str = 'dirichlet',
                left_temp: float = 0.0, right_temp: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the complete heat diffusion simulation.
        
        Parameters:
        -----------
        T_initial : np.ndarray
            Initial temperature distribution
        bc_type : str, optional
            Type of boundary condition ('dirichlet' or 'neumann', default: 'dirichlet')
        left_temp : float, optional
            Left boundary temperature for Dirichlet BC (default: 0.0)
        right_temp : float, optional
            Right boundary temperature for Dirichlet BC (default: 0.0)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (x, t, T_history) - spatial grid, time array, and temperature history
        """
        print(f"Starting heat diffusion simulation...")
        print(f"  Boundary condition: {bc_type}")
        
        # Check stability condition
        if self.dt > self.dt_max:
            warnings.warn(f"Time step {self.dt:.6f} s exceeds maximum stable step {self.dt_max:.6f} s. "
                         f"Simulation may be unstable!")
        
        # Set initial condition
        self.set_initial_condition(T_initial)
        
        # Apply initial boundary conditions
        self.apply_boundary_conditions(bc_type, left_temp, right_temp)
        self.T_history[0] = self.T.copy()
        
        # Time stepping loop
        for n in range(1, self.nt + 1):
            # Perform FTCS step
            self.ftcs_step()
            
            # Apply boundary conditions
            self.apply_boundary_conditions(bc_type, left_temp, right_temp)
            
            # Store temperature history
            self.T_history[n] = self.T.copy()
            
            # Progress indicator
            if n % max(1, self.nt // 10) == 0:
                progress = 100 * n / self.nt
                print(f"  Progress: {progress:.1f}%")
        
        print("Simulation completed!")
        print(f"  Final temperature range: [{self.T.min():.3f}, {self.T.max():.3f}] K")
        
        return self.x, self.t, self.T_history
    
    def get_final_temperature(self) -> np.ndarray:
        """Get the final temperature distribution."""
        return self.T.copy()
    
    def get_energy(self) -> float:
        """
        Calculate the total thermal energy in the system.
        
        Returns:
        --------
        float
            Total thermal energy (proportional to sum of temperatures)
        """
        return np.sum(self.T) * self.dx
