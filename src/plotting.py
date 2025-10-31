"""
Plotting utilities for the heat diffusion simulator (Day 4).

Provides functions to save temperature profile plots and
temperature evolution snapshots without requiring a display.
"""

import matplotlib
# Use a non-interactive backend suitable for headless environments
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_temperature_profile(
    x: np.ndarray,
    T: np.ndarray,
    *,
    title: str = "Temperature Profile",
    xlabel: str = "Position (m)",
    ylabel: str = "Temperature (K)",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot and optionally save a single temperature profile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, T, "b-", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved profile plot: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_temperature_evolution(
    x: np.ndarray,
    t: np.ndarray,
    T_history: np.ndarray,
    *,
    n_snapshots: int = 10,
    title: str = "Temperature Evolution",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot and optionally save multiple snapshots of the temperature over time."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Choose snapshot indices
    idx = np.linspace(0, T_history.shape[0] - 1, max(2, n_snapshots), dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(idx)))
    for c, i in zip(colors, idx):
        ax.plot(x, T_history[i], color=c, linewidth=2, label=f"t={t[i]:.3f}s")

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved evolution plot: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
