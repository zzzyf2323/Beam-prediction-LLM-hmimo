"""Deterministic synthetic HMIMO channel generation for static evaluation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from hmimo.physics.fpws import (
    fpws_dictionaries_rectangular,
    propagating_modes_rectangular,
    upa_index_grid,
)


def _pair_to_full_index_map(nx: int, ny: int) -> Dict[Tuple[int, int], int]:
    gx, gy = upa_index_grid(nx, ny, flatten=True)
    return {(int(mx), int(my)): i for i, (mx, my) in enumerate(zip(gx, gy))}


def _prop_to_full_indices(pairs: np.ndarray, nx: int, ny: int) -> np.ndarray:
    mapping = _pair_to_full_index_map(nx, ny)
    return np.asarray([mapping[(int(mx), int(my))] for mx, my in pairs], dtype=int)


def build_propagating_fpws_bases(
    rx_dims: Tuple[int, int],
    tx_dims: Tuple[int, int],
    rx_mode_pairs: np.ndarray,
    tx_mode_pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build FPWS dictionary sub-bases restricted to propagating modal sets."""

    psi_r, psi_s = fpws_dictionaries_rectangular(rx_dims, tx_dims)
    rx_full_idx = _prop_to_full_indices(rx_mode_pairs, rx_dims[0], rx_dims[1])
    tx_full_idx = _prop_to_full_indices(tx_mode_pairs, tx_dims[0], tx_dims[1])
    return psi_r[:, rx_full_idx], psi_s[:, tx_full_idx]


def generate_static_hmimo_channel(config: Dict[str, object]) -> Dict[str, object]:
    """Generate a seed-controlled static HMIMO channel in modal and spatial domain.

    Returns a dictionary containing propagating-mode channel ``ha`` and spatial
    channel ``h`` reconstructed via FPWS dictionaries.
    """

    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)

    rx_dims = tuple(config["rx_dims"])
    tx_dims = tuple(config["tx_dims"])
    d_over_lambda = float(config.get("d_over_lambda", 0.5))

    rx_modes = propagating_modes_rectangular(rx_dims[0], rx_dims[1], d_over_lambda=d_over_lambda)
    tx_modes = propagating_modes_rectangular(tx_dims[0], tx_dims[1], d_over_lambda=d_over_lambda)

    nr = rx_modes.pairs.shape[0]
    ns = tx_modes.pairs.shape[0]

    n_clusters = int(config.get("n_clusters", 2))
    spread = float(config.get("cluster_spread", 0.8))

    rx_pairs = rx_modes.pairs.astype(float)
    tx_pairs = tx_modes.pairs.astype(float)

    ha = np.zeros((nr, ns), dtype=np.complex128)

    for _ in range(n_clusters):
        cr = int(rng.integers(0, nr))
        cs = int(rng.integers(0, ns))
        center_r = rx_pairs[cr]
        center_s = tx_pairs[cs]

        wr = np.exp(-np.sum((rx_pairs - center_r[None, :]) ** 2, axis=1) / (2.0 * spread * spread))
        ws = np.exp(-np.sum((tx_pairs - center_s[None, :]) ** 2, axis=1) / (2.0 * spread * spread))
        amp = (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)
        ha += amp * np.outer(wr, ws)

    ha /= max(np.linalg.norm(ha), 1e-12)

    psi_r_prop, psi_s_prop = build_propagating_fpws_bases(
        rx_dims=rx_dims,
        tx_dims=tx_dims,
        rx_mode_pairs=rx_modes.pairs,
        tx_mode_pairs=tx_modes.pairs,
    )
    h = psi_r_prop @ ha @ psi_s_prop.conj().T

    return {
        "ha": ha,
        "h": h,
        "rx_mode_pairs": rx_modes.pairs,
        "tx_mode_pairs": tx_modes.pairs,
        "rx_dims": rx_dims,
        "tx_dims": tx_dims,
    }
