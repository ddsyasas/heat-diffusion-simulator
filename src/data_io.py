"""
Data I/O utilities for the heat diffusion simulator (Day 4).

- Save simulation data to compressed NPZ
- Export data to CSV for external analysis
- Ensure output directories exist
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_simulation_npz(
    *,
    x: np.ndarray,
    t: np.ndarray,
    T_history: np.ndarray,
    parameters: Dict[str, Any],
    output_path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save simulation arrays and metadata to a compressed NPZ file."""
    directory = os.path.dirname(output_path)
    if directory:
        ensure_dir(directory)

    payload: Dict[str, Any] = {
        "x": x,
        "t": t,
        "T_history": T_history,
        "parameters": parameters,
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "nx": int(x.shape[0]),
            "nt": int(t.shape[0]),
            "dx": float(x[1] - x[0]) if x.shape[0] > 1 else 0.0,
            "dt": float(t[1] - t[0]) if t.shape[0] > 1 else 0.0,
        },
    }
    if extra:
        payload.update(extra)

    np.savez_compressed(output_path, **payload)
    print(f"Saved NPZ: {output_path}")


def export_csv(
    *,
    x: np.ndarray,
    t: np.ndarray,
    T_history: np.ndarray,
    output_path: str,
) -> None:
    """Export flattened time-position-temperature triplets to CSV."""
    directory = os.path.dirname(output_path)
    if directory:
        ensure_dir(directory)

    nt, nx = T_history.shape
    time_col = np.repeat(t, nx)
    pos_col = np.tile(x, nt)
    temp_col = T_history.reshape(-1)

    data = np.column_stack([time_col, pos_col, temp_col])
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="time,position,temperature",
        comments="",
        fmt="%.6f",
    )
    print(f"Saved CSV: {output_path}")
