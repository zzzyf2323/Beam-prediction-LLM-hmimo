"""FPWS helpers for rectangular UPA modal dictionaries.

This module provides compact utilities used by stage-1 HMIMO physics code:

* UPA index-grid helpers.
* Unitary FPWS-aligned transmit/receive dictionaries.
* Propagating-mode index sets for rectangular arrays.

The implementation is intentionally lightweight and configuration-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PropagatingModeSet:
    """Container for rectangular UPA propagating-mode indices.

    Attributes:
        mx: Centered x-axis mode indices.
        my: Centered y-axis mode indices.
        pairs: Array with shape ``(K, 2)`` where each row is ``(mx, my)``.
    """

    mx: np.ndarray
    my: np.ndarray
    pairs: np.ndarray


def centered_mode_indices(size: int) -> np.ndarray:
    """Return centered integer mode indices for a DFT grid.

    Example for ``size=4``: ``[-2, -1, 0, 1]``.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    start = -(size // 2)
    return np.arange(start, start + size, dtype=int)


def upa_index_grid(nx: int, ny: int, flatten: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Build rectangular UPA centered mode-index grids.

    Args:
        nx: Number of elements/modes on x-axis.
        ny: Number of elements/modes on y-axis.
        flatten: If ``True`` returns flattened vectors, otherwise 2-D meshgrids.

    Returns:
        Tuple ``(mx, my)`` with either flattened or 2-D grids.
    """

    mx = centered_mode_indices(nx)
    my = centered_mode_indices(ny)
    gx, gy = np.meshgrid(mx, my, indexing="xy")
    if flatten:
        return gx.reshape(-1), gy.reshape(-1)
    return gx, gy


def fpws_dictionary_1d(num_antennas: int) -> np.ndarray:
    """Create a unitary 1-D FPWS/DFT steering dictionary."""

    if num_antennas <= 0:
        raise ValueError("num_antennas must be positive")

    n = np.arange(num_antennas, dtype=float)
    m = centered_mode_indices(num_antennas).astype(float)
    phase = np.exp(-1j * 2.0 * np.pi * np.outer(n, m) / float(num_antennas))
    return phase / np.sqrt(float(num_antennas))


def fpws_dictionary_upa(nx: int, ny: int) -> np.ndarray:
    """Create a unitary 2-D FPWS dictionary for rectangular UPA.

    The resulting matrix maps modal coefficients (columns) to element-domain
    array samples (rows), with x-fast vectorization.
    """

    psi_x = fpws_dictionary_1d(nx)
    psi_y = fpws_dictionary_1d(ny)
    return np.kron(psi_y, psi_x)


def fpws_dictionaries_rectangular(
    nrx: Tuple[int, int], ntx: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return receive/transmit FPWS dictionaries ``(Psi_R, Psi_S)``.

    Args:
        nrx: Receive UPA dimensions ``(Nx_r, Ny_r)``.
        ntx: Transmit UPA dimensions ``(Nx_s, Ny_s)``.
    """

    psi_r = fpws_dictionary_upa(nrx[0], nrx[1])
    psi_s = fpws_dictionary_upa(ntx[0], ntx[1])
    return psi_r, psi_s


def propagating_modes_rectangular(
    nx: int,
    ny: int,
    d_over_lambda: float = 0.5,
) -> PropagatingModeSet:
    """Return propagating-mode indices for a rectangular UPA modal grid.

    A mode pair ``(mx, my)`` is considered propagating if

    ``(mx/(nx*d/lambda))^2 + (my/(ny*d/lambda))^2 <= 1``.
    """

    if d_over_lambda <= 0:
        raise ValueError("d_over_lambda must be positive")

    gx, gy = upa_index_grid(nx, ny, flatten=True)
    ux = gx / (float(nx) * d_over_lambda)
    uy = gy / (float(ny) * d_over_lambda)
    mask = (ux**2 + uy**2) <= 1.0 + 1e-12
    pairs = np.stack([gx[mask], gy[mask]], axis=1)
    return PropagatingModeSet(mx=gx[mask], my=gy[mask], pairs=pairs)
