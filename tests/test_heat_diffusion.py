"""
Unit tests for the heat diffusion simulator (Day 6).

Comprehensive tests for core functionality, physics accuracy,
and edge cases.
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from heat_diffusion import HeatDiffusionSimulator
from initial_conditions import gaussian, step_function, get_initial_condition


class TestHeatDiffusionSimulator:
    """Test cases for HeatDiffusionSimulator class."""
    
    def test_initialization(self):
        """Test simulator initialization."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=100, t_end=10.0)
        
        assert simulator.alpha == 0.01
        assert simulator.L == 1.0
        assert simulator.nx == 100
        assert simulator.t_end == 10.0
        assert len(simulator.x) == 100
        assert simulator.x[0] == 0.0
        assert simulator.x[-1] == 1.0
    
    def test_spatial_grid(self):
        """Test spatial grid creation."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=2.0, nx=50, t_end=5.0)
        
        assert simulator.dx == pytest.approx(2.0 / 49, rel=1e-6)
        assert len(simulator.x) == 50
        assert simulator.x.max() == 2.0
    
    def test_stability_condition(self):
        """Test CFL stability condition."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=100, t_end=10.0)
        
        # Check that dt satisfies stability condition
        assert simulator.dt <= simulator.dt_max
        assert simulator.dt_max == pytest.approx(0.5 * simulator.dx**2 / simulator.alpha)
    
    def test_dirichlet_boundary_conditions(self):
        """Test Dirichlet boundary conditions."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=50, t_end=1.0)
        simulator.T = np.ones(50)
        
        simulator.apply_boundary_conditions('dirichlet', left_temp=0.0, right_temp=0.5)
        
        assert simulator.T[0] == 0.0
        assert simulator.T[-1] == 0.5
        # Interior points unchanged
        assert np.allclose(simulator.T[1:-1], 1.0)
    
    def test_neumann_boundary_conditions(self):
        """Test Neumann (insulated) boundary conditions."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=50, t_end=1.0)
        simulator.T = np.linspace(0, 1, 50)
        
        simulator.apply_boundary_conditions('neumann')
        
        # Neumann: T[0] = T[1] and T[-1] = T[-2]
        assert simulator.T[0] == simulator.T[1]
        assert simulator.T[-1] == simulator.T[-2]
    
    def test_invalid_boundary_condition(self):
        """Test error handling for invalid boundary conditions."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=50, t_end=1.0)
        
        with pytest.raises(ValueError, match="Unknown boundary condition"):
            simulator.apply_boundary_conditions('invalid_type')
    
    def test_gaussian_diffusion(self):
        """Test that Gaussian peak diffuses correctly."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=100, t_end=2.0)
        
        # Create Gaussian initial condition
        ic_func = get_initial_condition('gaussian', center=0.5, width=0.1, amplitude=1.0)
        T_initial = ic_func(simulator.x)
        
        x, t, T_history = simulator.simulate(
            T_initial=T_initial,
            bc_type='dirichlet',
            left_temp=0.0,
            right_temp=0.0
        )
        
        # Check that peak decreases (heat spreads)
        initial_max = T_history[0].max()
        final_max = T_history[-1].max()
        assert final_max < initial_max
        
        # Check boundaries remain fixed
        assert np.allclose(T_history[:, 0], 0.0)
        assert np.allclose(T_history[:, -1], 0.0)
    
    def test_energy_conservation_neumann(self):
        """Test energy conservation with Neumann BC."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=50, t_end=1.0)
        
        # Gaussian IC
        ic_func = get_initial_condition('gaussian', center=0.5, width=0.1, amplitude=1.0)
        T_initial = ic_func(simulator.x)
        
        x, t, T_history = simulator.simulate(
            T_initial=T_initial,
            bc_type='neumann',  # Insulated - should conserve energy
            left_temp=0.0,
            right_temp=0.0
        )
        
        # Calculate energy at start and end
        energy_initial = np.sum(T_history[0]) * simulator.dx
        energy_final = np.sum(T_history[-1]) * simulator.dx
        
        # Energy should be approximately conserved with Neumann BC
        energy_change = abs(energy_final - energy_initial) / energy_initial
        assert energy_change < 0.01  # Less than 1% change
    
    def test_step_function_smoothing(self):
        """Test that step function smooths over time."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=100, t_end=5.0)
        
        # Step function IC
        ic_func = get_initial_condition('step', position=0.5, left_temp=0.0, right_temp=1.0)
        T_initial = ic_func(simulator.x)
        
        x, t, T_history = simulator.simulate(
            T_initial=T_initial,
            bc_type='dirichlet',
            left_temp=0.0,
            right_temp=1.0
        )
        
        # Check that discontinuity smooths out
        # Measure gradient at center
        center_idx = len(x) // 2
        initial_gradient = abs(T_history[0, center_idx + 1] - T_history[0, center_idx - 1])
        final_gradient = abs(T_history[-1, center_idx + 1] - T_history[-1, center_idx - 1])
        
        assert final_gradient < initial_gradient


class TestInitialConditions:
    """Test cases for initial condition functions."""
    
    def test_gaussian_center(self):
        """Test Gaussian IC centered correctly."""
        x = np.linspace(0, 1, 100)
        T = gaussian(x, center=0.5, width=0.1, amplitude=1.0)
        
        # Maximum should be at center
        max_idx = np.argmax(T)
        assert x[max_idx] == pytest.approx(0.5, abs=0.02)
        assert T.max() == pytest.approx(1.0)
    
    def test_gaussian_default_center(self):
        """Test Gaussian IC with default center."""
        x = np.linspace(0, 2, 100)
        T = gaussian(x)  # Should center at 1.0
        
        max_idx = np.argmax(T)
        assert x[max_idx] == pytest.approx(1.0, abs=0.04)
    
    def test_step_function(self):
        """Test step function IC."""
        x = np.linspace(0, 1, 100)
        T = step_function(x, position=0.5, left_temp=0.0, right_temp=1.0)
        
        # Check left side
        assert np.allclose(T[x < 0.5], 0.0)
        # Check right side
        assert np.allclose(T[x >= 0.5], 1.0)
    
    def test_get_initial_condition_gaussian(self):
        """Test IC factory for Gaussian."""
        ic_func = get_initial_condition('gaussian', center=0.5, width=0.1)
        x = np.linspace(0, 1, 50)
        T = ic_func(x)
        
        assert len(T) == 50
        assert T.max() <= 1.0
    
    def test_get_initial_condition_step(self):
        """Test IC factory for step function."""
        ic_func = get_initial_condition('step', position=0.5, left_temp=0.0, right_temp=1.0)
        x = np.linspace(0, 1, 50)
        T = ic_func(x)
        
        assert len(T) == 50
        assert T.min() == 0.0
        assert T.max() == 1.0
    
    def test_invalid_ic_type(self):
        """Test error for invalid IC type."""
        with pytest.raises(ValueError, match="Unknown initial condition"):
            get_initial_condition('invalid_type')


class TestPhysics:
    """Test physical behavior of heat diffusion."""
    
    def test_heat_flows_hot_to_cold(self):
        """Test that heat flows from hot to cold regions."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=50, t_end=1.0)
        
        # Hot in middle, cold at ends
        T_initial = np.zeros(50)
        T_initial[20:30] = 1.0
        
        x, t, T_history = simulator.simulate(
            T_initial=T_initial,
            bc_type='dirichlet',
            left_temp=0.0,
            right_temp=0.0
        )
        
        # Temperature at hot region should decrease
        assert T_history[-1, 25] < T_history[0, 25]
        
        # Temperature at cold regions should increase
        assert T_history[-1, 10] > T_history[0, 10]
        assert T_history[-1, 40] > T_history[0, 40]
    
    def test_steady_state_dirichlet(self):
        """Test approach to steady state with Dirichlet BC."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=50, t_end=20.0)
        
        # Start with uniform temperature
        T_initial = np.ones(50) * 0.5
        
        x, t, T_history = simulator.simulate(
            T_initial=T_initial,
            bc_type='dirichlet',
            left_temp=0.0,
            right_temp=1.0
        )
        
        # Should approach linear profile
        expected_steady = np.linspace(0, 1, 50)
        
        # Final state should be close to linear
        assert np.allclose(T_history[-1], expected_steady, atol=0.1)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_grid(self):
        """Test with minimum grid size."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=3, t_end=0.1)
        
        assert simulator.nx == 3
        assert len(simulator.x) == 3
    
    def test_large_grid(self):
        """Test with large grid."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=500, t_end=1.0)
        
        # Should complete without errors
        assert simulator.nx == 500
        # Time step should be very small
        assert simulator.dt < 0.001
    
    def test_zero_initial_condition(self):
        """Test with zero initial temperature."""
        simulator = HeatDiffusionSimulator(alpha=0.01, L=1.0, nx=50, t_end=1.0)
        
        T_initial = np.zeros(50)
        
        x, t, T_history = simulator.simulate(
            T_initial=T_initial,
            bc_type='dirichlet',
            left_temp=0.0,
            right_temp=0.0
        )
        
        # Should remain zero
        assert np.allclose(T_history[-1], 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

