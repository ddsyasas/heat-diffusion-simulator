"""
Initial condition functions for the heat diffusion simulator.

Day 3-6: Provide various initial conditions for simulations.
Includes:
- Gaussian peak centered in the domain or at a specified position
- Step function separating left/right temperatures at a position
- Sine and cosine waves (Day 6)
- Linear temperature gradient (Day 6)
- Constant temperature (Day 6)
"""

import numpy as np
from typing import Callable


def gaussian(x: np.ndarray, *, center: float | None = None, width: float | None = None, amplitude: float = 1.0) -> np.ndarray:
    """
    Gaussian initial temperature distribution.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    center : float | None
        Center of the Gaussian. Defaults to the middle of the domain when None.
    width : float | None
        Standard deviation. Defaults to 10% of domain length when None.
    amplitude : float
        Peak amplitude.
    """
    if center is None:
        center = 0.5 * (x.min() + x.max())
    if width is None:
        width = 0.1 * (x.max() - x.min())
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def step_function(x: np.ndarray, *, position: float | None = None, left_temp: float = 0.0, right_temp: float = 1.0) -> np.ndarray:
    """
    Step function initial temperature distribution.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    position : float | None
        Position of the step. Defaults to the middle of the domain when None.
    left_temp : float
        Temperature to the left of the step.
    right_temp : float
        Temperature to the right of the step.
    """
    if position is None:
        position = 0.5 * (x.min() + x.max())
    T = np.empty_like(x)
    T[:] = right_temp
    T[x < position] = left_temp
    return T


def sine_wave(x: np.ndarray, *, wavelength: float | None = None, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """
    Sine wave initial temperature distribution.
    
    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    wavelength : float | None
        Wavelength. Defaults to domain length when None.
    amplitude : float
        Amplitude (default: 1.0).
    phase : float
        Phase shift in radians (default: 0.0).
    """
    if wavelength is None:
        wavelength = x.max() - x.min()
    return amplitude * np.sin(2 * np.pi * x / wavelength + phase)


def cosine_wave(x: np.ndarray, *, wavelength: float | None = None, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """
    Cosine wave initial temperature distribution.
    
    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    wavelength : float | None
        Wavelength. Defaults to domain length when None.
    amplitude : float
        Amplitude (default: 1.0).
    phase : float
        Phase shift in radians (default: 0.0).
    """
    if wavelength is None:
        wavelength = x.max() - x.min()
    return amplitude * np.cos(2 * np.pi * x / wavelength + phase)


def linear(x: np.ndarray, *, left_temp: float = 0.0, right_temp: float = 1.0) -> np.ndarray:
    """
    Linear temperature gradient.
    
    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    left_temp : float
        Temperature at left boundary (default: 0.0).
    right_temp : float
        Temperature at right boundary (default: 1.0).
    """
    return left_temp + (right_temp - left_temp) * (x - x.min()) / (x.max() - x.min())


def constant(x: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """
    Constant temperature distribution.
    
    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    temperature : float
        Constant temperature value (default: 1.0).
    """
    return np.full_like(x, temperature)


def get_initial_condition(ic_type: str, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory for initial condition functions.

    Parameters
    ----------
    ic_type : str
        One of {'gaussian', 'step', 'sine', 'cosine', 'linear', 'constant'}.
    kwargs : dict
        Parameters forwarded to the specific IC function.
    """
    ic_type_l = ic_type.lower()
    if ic_type_l == "gaussian":
        return lambda x: gaussian(x, **kwargs)
    if ic_type_l in ("step", "step_function"):
        return lambda x: step_function(x, **kwargs)
    if ic_type_l in ("sine", "sine_wave"):
        return lambda x: sine_wave(x, **kwargs)
    if ic_type_l in ("cosine", "cosine_wave"):
        return lambda x: cosine_wave(x, **kwargs)
    if ic_type_l == "linear":
        return lambda x: linear(x, **kwargs)
    if ic_type_l == "constant":
        return lambda x: constant(x, **kwargs)
    raise ValueError(
        f"Unknown initial condition type: {ic_type}. "
        f"Available: ['gaussian', 'step', 'sine', 'cosine', 'linear', 'constant']"
    )
