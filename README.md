# Heat Diffusion Simulator

A Python project for simulating heat diffusion in 1D materials.

## Project Overview

This project aims to implement a numerical simulation of heat diffusion using the heat equation:

```
∂T/∂t = α * ∂²T/∂x²
```

where:
- T is temperature
- t is time
- x is position  
- α is thermal diffusivity

## Project Structure

```
Heat Diffusion Simulator/
├── src/                    # Source code
├── tests/                  # Unit tests
├── results/                # Output results
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- pytest (for testing)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

Run simulations using the command-line interface:

```bash
cd src
python main.py --alpha 0.01 --L 1.0 --nx 100 --t-end 10.0 --ic gaussian
```

### Available Options

**Required Parameters:**
- `--alpha`: Thermal diffusivity (m²/s)
- `--L`: Length of material (m)
- `--nx`: Number of grid points
- `--t-end`: Simulation time (s)

**Initial Conditions:**
- `--ic {gaussian,step}`: Initial condition type
- `--ic-center`: Center position (default: middle)
- `--ic-width`: Width for Gaussian IC
- `--ic-amplitude`: Amplitude (default: 1.0)

**Boundary Conditions:**
- `--bc {dirichlet,neumann}`: Boundary condition type
- `--left-temp`: Left boundary temperature
- `--right-temp`: Right boundary temperature

**Output Options:**
- `--output`: Output directory (default: output)
- `--csv`: Export to CSV format
- `--no-plot`: Skip plots
- `--no-save`: Skip saving data
- `--quiet`: Suppress output

### Examples

**Gaussian heat pulse:**
```bash
python main.py --alpha 0.01 --L 1.0 --nx 100 --t-end 10.0 --ic gaussian
```

**Step function with Neumann boundaries:**
```bash
python main.py --alpha 0.005 --L 2.0 --nx 200 --t-end 20.0 --ic step --bc neumann --csv
```

**Custom output directory:**
```bash
python main.py --alpha 0.01 --L 1.0 --nx 150 --t-end 15.0 --ic gaussian --output my_results
```

## Features

- ✅ FTCS (Forward-Time Central-Space) finite difference method
- ✅ Multiple initial conditions (Gaussian, step function)
- ✅ Boundary conditions (Dirichlet, Neumann)
- ✅ Stability checking (CFL condition)
- ✅ Temperature profile visualization
- ✅ Temperature evolution plots
- ✅ Data export (NPZ, CSV)
- ✅ Command-line interface

## Development Status

This project is being developed incrementally:
- Day 1: Project setup ✓
- Day 2: Core physics implementation ✓
- Day 3: Initial and boundary conditions ✓
- Day 4: Visualization and data output ✓
- Day 5: Command-line interface ✓

## Author

Student - Applied Physics Course
