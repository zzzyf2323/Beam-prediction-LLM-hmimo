"""Deterministic dynamic HMIMO channel sequence generation in FPWS domain."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from hmimo.simulation.channel import build_propagating_fpws_bases
from hmimo.physics.fpws import propagating_modes_rectangular


def generate_dynamic_hmimo_sequence(config: Dict[str, object]) -> Dict[str, object]:
    """Generate a short time-correlated HMIMO sequence.

    The sequence is formed by slowly drifting cluster centers and AR-smoothed
    complex amplitudes in the propagating modal domain.
    """

    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)

    rx_dims = tuple(config["rx_dims"])
    tx_dims = tuple(config["tx_dims"])
    n_frames = int(config.get("n_frames", 5))
    n_clusters = int(config.get("n_clusters", 2))
    drift_std = float(config.get("drift_std", 0.15))
    amp_rho = float(config.get("amp_rho", 0.9))
    d_over_lambda = float(config.get("d_over_lambda", 0.5))
    spread = float(config.get("cluster_spread", 0.9))

    rx_modes = propagating_modes_rectangular(rx_dims[0], rx_dims[1], d_over_lambda=d_over_lambda)
    tx_modes = propagating_modes_rectangular(tx_dims[0], tx_dims[1], d_over_lambda=d_over_lambda)
    rx_pairs = rx_modes.pairs.astype(float)
    tx_pairs = tx_modes.pairs.astype(float)

    nr = rx_pairs.shape[0]
    ns = tx_pairs.shape[0]

    psi_r_prop, psi_s_prop = build_propagating_fpws_bases(
        rx_dims=rx_dims,
        tx_dims=tx_dims,
        rx_mode_pairs=rx_modes.pairs,
        tx_mode_pairs=tx_modes.pairs,
    )

    # Initialize cluster states.
    rx_centers = rx_pairs[rng.integers(0, nr, size=n_clusters)].copy()
    tx_centers = tx_pairs[rng.integers(0, ns, size=n_clusters)].copy()
    amps = (rng.normal(size=n_clusters) + 1j * rng.normal(size=n_clusters)) / np.sqrt(2.0)

    ha_seq: List[np.ndarray] = []
    h_seq: List[np.ndarray] = []

    for _ in range(n_frames):
        ha = np.zeros((nr, ns), dtype=np.complex128)
        for k in range(n_clusters):
            wr = np.exp(-np.sum((rx_pairs - rx_centers[k][None, :]) ** 2, axis=1) / (2.0 * spread * spread))
            ws = np.exp(-np.sum((tx_pairs - tx_centers[k][None, :]) ** 2, axis=1) / (2.0 * spread * spread))
            ha += amps[k] * np.outer(wr, ws)

        ha /= max(np.linalg.norm(ha), 1e-12)
        h = psi_r_prop @ ha @ psi_s_prop.conj().T

        ha_seq.append(ha)
        h_seq.append(h)

        # Smooth temporal evolution for next frame.
        rx_centers += drift_std * rng.normal(size=rx_centers.shape)
        tx_centers += drift_std * rng.normal(size=tx_centers.shape)
        amps = amp_rho * amps + np.sqrt(max(1.0 - amp_rho**2, 0.0)) * (
            rng.normal(size=n_clusters) + 1j * rng.normal(size=n_clusters)
        ) / np.sqrt(2.0)

    return {
        "ha_seq": ha_seq,
        "h_seq": h_seq,
        "rx_mode_pairs": rx_modes.pairs,
        "tx_mode_pairs": tx_modes.pairs,
        "rx_dims": rx_dims,
        "tx_dims": tx_dims,
    }
