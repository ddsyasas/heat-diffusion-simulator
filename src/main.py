#!/usr/bin/env python3
"""
Command-line interface for the Heat Diffusion Simulator (Day 5).

This script provides a user-friendly CLI to run heat diffusion simulations
with various parameters and output options.
"""

import argparse
import sys
import os
import contextlib
import io

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heat_diffusion import HeatDiffusionSimulator
from initial_conditions import get_initial_condition
from plotting import plot_temperature_profile, plot_temperature_evolution
from data_io import save_simulation_npz, export_csv, ensure_dir


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Heat Diffusion Simulator - Simulate heat diffusion in 1D materials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation with Gaussian initial condition
  python main.py --alpha 0.01 --L 1.0 --nx 100 --t-end 10.0 --ic gaussian

  # Step function with Neumann boundaries
  python main.py --alpha 0.005 --L 2.0 --nx 200 --t-end 20.0 --ic step --bc neumann

  # Save outputs to specific directory
  python main.py --alpha 0.01 --L 1.0 --nx 150 --t-end 15.0 --ic gaussian --output results/my_sim
        """
    )
    
    # Required simulation parameters
    parser.add_argument('--alpha', type=float, required=True,
                       help='Thermal diffusivity of the material (m²/s)')
    parser.add_argument('--L', type=float, required=True,
                       help='Total length of the material (m)')
    parser.add_argument('--nx', type=int, required=True,
                       help='Number of grid points along the rod')
    parser.add_argument('--t-end', type=float, required=True,
                       help='Total simulation time (s)')
    
    # Initial condition options
    parser.add_argument('--ic', type=str, default='gaussian',
                       choices=['gaussian', 'step'],
                       help='Initial condition type (default: gaussian)')
    parser.add_argument('--ic-center', type=float, default=None,
                       help='Center position for initial condition (default: middle)')
    parser.add_argument('--ic-width', type=float, default=None,
                       help='Width parameter for Gaussian IC (default: 10%% of L)')
    parser.add_argument('--ic-amplitude', type=float, default=1.0,
                       help='Amplitude for initial condition (default: 1.0)')
    
    # Boundary condition options
    parser.add_argument('--bc', type=str, default='dirichlet',
                       choices=['dirichlet', 'neumann'],
                       help='Boundary condition type (default: dirichlet)')
    parser.add_argument('--left-temp', type=float, default=0.0,
                       help='Left boundary temperature for Dirichlet BC (default: 0.0)')
    parser.add_argument('--right-temp', type=float, default=0.0,
                       help='Right boundary temperature for Dirichlet BC (default: 0.0)')
    
    # Output options
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving data files')
    parser.add_argument('--csv', action='store_true',
                       help='Export data to CSV format')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress simulation output')
    
    return parser.parse_args()


def validate_parameters(args):
    """Validate input parameters."""
    errors = []
    
    if args.alpha <= 0:
        errors.append("alpha must be positive")
    if args.L <= 0:
        errors.append("L must be positive")
    if args.nx < 3:
        errors.append("nx must be at least 3")
    if args.t_end <= 0:
        errors.append("t_end must be positive")
    if args.ic_amplitude < 0:
        errors.append("ic_amplitude must be non-negative")
    
    if errors:
        print("Parameter validation errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


def create_initial_condition(args):
    """Create initial condition function based on arguments."""
    ic_kwargs = {}
    
    if args.ic == 'gaussian':
        ic_kwargs['amplitude'] = args.ic_amplitude
        if args.ic_center is not None:
            ic_kwargs['center'] = args.ic_center
        if args.ic_width is not None:
            ic_kwargs['width'] = args.ic_width
    
    elif args.ic == 'step':
        ic_kwargs['left_temp'] = 0.0
        ic_kwargs['right_temp'] = args.ic_amplitude
        if args.ic_center is not None:
            ic_kwargs['position'] = args.ic_center
    
    return get_initial_condition(args.ic, **ic_kwargs)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate parameters
    validate_parameters(args)
    
    # Suppress output if quiet mode
    if args.quiet:
        # Simple way to suppress prints from simulator
        f = io.StringIO()
        context = contextlib.redirect_stdout(f)
    else:
        context = contextlib.nullcontext()
    
    print(f"Heat Diffusion Simulator - Day 5")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  α = {args.alpha} m²/s")
    print(f"  L = {args.L} m")
    print(f"  nx = {args.nx}")
    print(f"  t_end = {args.t_end} s")
    print(f"  IC = {args.ic}")
    print(f"  BC = {args.bc}")
    print()
    
    # Create simulator
    with context:
        simulator = HeatDiffusionSimulator(
            alpha=args.alpha,
            L=args.L,
            nx=args.nx,
            t_end=args.t_end
        )
    
    # Create initial condition
    ic_func = create_initial_condition(args)
    T_initial = ic_func(simulator.x)
    
    # Run simulation
    print("Running simulation...")
    with context:
        x, t, T_history = simulator.simulate(
            T_initial=T_initial,
            bc_type=args.bc,
            left_temp=args.left_temp,
            right_temp=args.right_temp
        )
    
    print(f"Simulation complete!")
    print(f"  Final temperature range: [{T_history[-1].min():.3f}, {T_history[-1].max():.3f}] K")
    
    # Create output directory
    ensure_dir(args.output)
    
    # Generate plots
    if not args.no_plot:
        print("\nGenerating plots...")
        plot_temperature_profile(
            x, T_history[-1],
            title=f"Final Temperature Profile (t={t[-1]:.3f}s)",
            save_path=os.path.join(args.output, "profile.png"),
            show=False
        )
        plot_temperature_evolution(
            x, t, T_history,
            n_snapshots=10,
            title="Temperature Evolution",
            save_path=os.path.join(args.output, "evolution.png"),
            show=False
        )
    
    # Save data
    if not args.no_save:
        print("Saving data...")
        parameters = {
            'alpha': args.alpha,
            'L': args.L,
            'nx': args.nx,
            't_end': args.t_end,
            'ic': args.ic,
            'bc': args.bc,
            'left_temp': args.left_temp,
            'right_temp': args.right_temp,
            'dx': simulator.dx,
            'dt': simulator.dt,
        }
        
        save_simulation_npz(
            x=x, t=t, T_history=T_history,
            parameters=parameters,
            output_path=os.path.join(args.output, "simulation.npz")
        )
        
        if args.csv:
            export_csv(
                x=x, t=t, T_history=T_history,
                output_path=os.path.join(args.output, "simulation.csv")
            )
    
    print(f"\n✅ Done! Results saved to: {args.output}")


if __name__ == "__main__":
    main()

