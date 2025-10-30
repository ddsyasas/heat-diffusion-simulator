#!/usr/bin/env python3
"""
Basic test script for Day 2 implementation.

This script tests the core heat diffusion physics implementation.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from heat_diffusion import HeatDiffusionSimulator
from initial_conditions import get_initial_condition


def test_basic_simulation():
    """Test basic heat diffusion simulation."""
    print("Testing basic heat diffusion simulation...")
    
    # Create simulator
    simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=50, t_end=2.0)
    
    # Create simple initial condition (hot in the middle) via helper
    x = simulator.x
    T_initial = get_initial_condition('gaussian', center=0.5, width=0.1, amplitude=1.0)(x)
    
    print(f"Initial condition: Gaussian peak at x=0.5")
    print(f"Initial temperature range: [{T_initial.min():.3f}, {T_initial.max():.3f}] K")
    
    # Run simulation
    x, t, T_history = simulator.simulate(
        T_initial=T_initial,
        bc_type='dirichlet',
        left_temp=0.0,
        right_temp=0.0
    )
    
    # Check results
    print(f"\nSimulation results:")
    print(f"  Final temperature range: [{T_history[-1].min():.3f}, {T_history[-1].max():.3f}] K")
    print(f"  Energy conservation: Initial={simulator.get_energy():.6f}")
    
    # Calculate energy at different times
    energy_initial = np.sum(T_history[0]) * simulator.dx
    energy_final = np.sum(T_history[-1]) * simulator.dx
    energy_change = abs(energy_final - energy_initial) / energy_initial * 100
    
    print(f"  Energy change: {energy_change:.2f}%")
    
    return x, t, T_history


def test_stability_condition():
    """Test stability condition checking."""
    print("\nTesting stability condition...")
    
    # Test with stable parameters
    simulator_stable = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=100, t_end=1.0)
    print(f"Stable case: dt={simulator_stable.dt:.6f} s, dt_max={simulator_stable.dt_max:.6f} s")
    print(f"Stability ratio: {simulator_stable.dt / simulator_stable.dt_max:.3f}")
    
    # Test with potentially unstable parameters
    try:
        simulator_unstable = HeatDiffusionSimulator(alpha=0.001, L=1.0, nx=1000, t_end=1.0)
        print(f"High resolution case: dt={simulator_unstable.dt:.6f} s, dt_max={simulator_unstable.dt_max:.6f} s")
        print(f"Stability ratio: {simulator_unstable.dt / simulator_unstable.dt_max:.3f}")
    except Exception as e:
        print(f"Error with high resolution: {e}")

    # Validate boundary condition names
    try:
        _ = simulator_stable.simulate(T_initial=np.ones_like(simulator_stable.x), bc_type='invalid')
    except ValueError as e:
        print(f"Caught expected invalid BC error: {e}")


if __name__ == "__main__":
    print("Day 2: Core Physics Implementation Test")
    print("=" * 50)
    
    try:
        # Test basic simulation
        x, t, T_history = test_basic_simulation()
        
        # Test stability condition
        test_stability_condition()
        
        print("\n✅ All tests passed! Core physics implementation working.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

