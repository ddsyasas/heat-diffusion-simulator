"""
Numba-optimized functions for the heat diffusion simulator (Day 6).

This module provides JIT-compiled functions for improved performance.
Uses Numba to compile critical loops to machine code.
"""

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: no-op decorator
    def njit(func):
        return func

import numpy as np


@njit
def ftcs_step_optimized(T: np.ndarray, alpha: float, dt: float, dx: float) -> np.ndarray:
    """
    Numba-optimized FTCS time step.
    
    Parameters
    ----------
    T : np.ndarray
        Current temperature array.
    alpha : float
        Thermal diffusivity.
    dt : float
        Time step.
    dx : float
        Spatial step.
        
    Returns
    -------
    np.ndarray
        Updated temperature array.
    """
    nx = len(T)
    T_new = np.empty_like(T)
    
    # Copy boundary values
    T_new[0] = T[0]
    T_new[-1] = T[-1]
    
    # FTCS scheme for interior points
    factor = alpha * dt / (dx * dx)
    for i in range(1, nx - 1):
        T_new[i] = T[i] + factor * (T[i+1] - 2*T[i] + T[i-1])
    
    return T_new


@njit
def calculate_energy_optimized(T: np.ndarray, dx: float) -> float:
    """
    Numba-optimized energy calculation.
    
    Parameters
    ----------
    T : np.ndarray
        Temperature array.
    dx : float
        Spatial step.
        
    Returns
    -------
    float
        Total energy.
    """
    return np.sum(T) * dx


@njit
def apply_dirichlet_bc_optimized(T: np.ndarray, left_temp: float, right_temp: float) -> None:
    """
    Numba-optimized Dirichlet boundary condition application.
    
    Parameters
    ----------
    T : np.ndarray
        Temperature array (modified in place).
    left_temp : float
        Left boundary temperature.
    right_temp : float
        Right boundary temperature.
    """
    T[0] = left_temp
    T[-1] = right_temp


@njit
def apply_neumann_bc_optimized(T: np.ndarray) -> None:
    """
    Numba-optimized Neumann boundary condition application.
    
    Parameters
    ----------
    T : np.ndarray
        Temperature array (modified in place).
    """
    T[0] = T[1]
    T[-1] = T[-2]


def get_performance_info() -> dict:
    """
    Get information about optimization availability.
    
    Returns
    -------
    dict
        Dictionary with performance information.
    """
    return {
        'numba_available': NUMBA_AVAILABLE,
        'optimization_enabled': NUMBA_AVAILABLE,
        'version': 'optimized' if NUMBA_AVAILABLE else 'standard'
    }


# Usage example
if __name__ == "__main__":
    print("Numba Optimization Module")
    print("=" * 40)
    info = get_performance_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    if NUMBA_AVAILABLE:
        print("\n✓ Numba optimization available!")
        print("  Use optimized functions for better performance.")
    else:
        print("\n⚠ Numba not available.")
        print("  Install with: pip install numba")

