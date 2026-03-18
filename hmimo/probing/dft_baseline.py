"""DFT probing baseline context builders in modal-operator form."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from hmimo.physics.fpws import centered_mode_indices


def _dft_1d(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=float)
    phase = np.exp(-1j * 2.0 * np.pi * np.outer(idx, idx) / float(n))
    return phase / np.sqrt(float(n))


def _dft_2d(nx: int, ny: int) -> np.ndarray:
    return np.kron(_dft_1d(ny), _dft_1d(nx))


def _grid_pairs(nx: int, ny: int) -> np.ndarray:
    mx = centered_mode_indices(nx)
    my = centered_mode_indices(ny)
    gx, gy = np.meshgrid(mx, my, indexing="xy")
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)


def _nearest_indices(source_pairs: np.ndarray, target_pairs: np.ndarray) -> np.ndarray:
    picked = []
    used: set[int] = set()
    for p in source_pairs:
        d = np.sum((target_pairs - p[None, :]) ** 2, axis=1)
        order = np.argsort(d)
        chosen = None
        for idx in order.tolist():
            if idx not in used:
                chosen = idx
                break
        if chosen is None:
            chosen = int(order[0])
        picked.append(chosen)
        used.add(chosen)
    return np.asarray(picked, dtype=int)


def build_dft_baseline_contexts(
    rx_dims: Tuple[int, int],
    tx_dims: Tuple[int, int],
    rx_mode_pairs: np.ndarray,
    tx_mode_pairs: np.ndarray,
    n_slots: int,
    n_rx_rf: int,
    n_tx_rf: int,
    seed: int = 0,
    no_repeat: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """Build DFT-baseline contexts matched to modal operator interface."""

    rng = np.random.default_rng(seed)
    rx_pairs = np.asarray(rx_mode_pairs, dtype=int)
    tx_pairs = np.asarray(tx_mode_pairs, dtype=int)

    full_rx_pairs = _grid_pairs(rx_dims[0], rx_dims[1])
    full_tx_pairs = _grid_pairs(tx_dims[0], tx_dims[1])

    keep_rx = _nearest_indices(rx_pairs, full_rx_pairs)
    keep_tx = _nearest_indices(tx_pairs, full_tx_pairs)

    c_rx = _dft_2d(rx_dims[0], rx_dims[1])
    c_tx = _dft_2d(tx_dims[0], tx_dims[1])

    u_rx = c_rx[np.ix_(keep_rx, keep_rx)]
    u_tx = c_tx[np.ix_(keep_tx, keep_tx)]

    nr = rx_pairs.shape[0]
    ns = tx_pairs.shape[0]
    rx_pool = np.arange(nr, dtype=int)
    tx_pool = np.arange(ns, dtype=int)
    used_rx: set[int] = set()
    used_tx: set[int] = set()

    contexts: List[Dict[str, np.ndarray]] = []
    for _ in range(n_slots):
        if no_repeat:
            avail_rx = [i for i in rx_pool.tolist() if i not in used_rx]
            avail_tx = [i for i in tx_pool.tolist() if i not in used_tx]
            pick_rx_pool = np.asarray(avail_rx if len(avail_rx) >= n_rx_rf else rx_pool, dtype=int)
            pick_tx_pool = np.asarray(avail_tx if len(avail_tx) >= n_tx_rf else tx_pool, dtype=int)
        else:
            pick_rx_pool = rx_pool
            pick_tx_pool = tx_pool

        rx_idx = np.asarray(rng.choice(pick_rx_pool, size=n_rx_rf, replace=False), dtype=int)
        tx_idx = np.asarray(rng.choice(pick_tx_pool, size=n_tx_rf, replace=False), dtype=int)
        used_rx.update(rx_idx.tolist())
        used_tx.update(tx_idx.tolist())

        s_rx = np.zeros((n_rx_rf, nr), dtype=np.complex128)
        s_rx[np.arange(n_rx_rf), rx_idx] = 1.0
        q = s_rx @ u_rx

        s_tx = np.zeros((n_tx_rf, ns), dtype=np.complex128)
        s_tx[np.arange(n_tx_rf), tx_idx] = 1.0
        p = (u_tx.conj().T @ s_tx.T)  # (ns, n_tx_rf)
        x = p @ np.ones((n_tx_rf,), dtype=np.complex128)

        contexts.append({"Q": q, "P": p, "x": x, "rx_indices": rx_idx, "tx_indices": tx_idx})

    return contexts
