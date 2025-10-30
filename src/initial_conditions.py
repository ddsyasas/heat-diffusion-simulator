"""
Initial condition functions for the heat diffusion simulator.

Day 3: Provide basic initial conditions used to start simulations.
Includes:
- Gaussian peak centered in the domain or at a specified position
- Step function separating left/right temperatures at a position
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


def get_initial_condition(ic_type: str, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory for initial condition functions.

    Parameters
    ----------
    ic_type : str
        One of {'gaussian', 'step'}.
    kwargs : dict
        Parameters forwarded to the specific IC function.
    """
    ic_type_l = ic_type.lower()
    if ic_type_l == "gaussian":
        return lambda x: gaussian(x, **kwargs)
    if ic_type_l in ("step", "step_function"):
        return lambda x: step_function(x, **kwargs)
    raise ValueError(
        f"Unknown initial condition type: {ic_type}. Available: ['gaussian', 'step']"
    )
