"""FPWS-aligned selection probing context builders."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _selection_matrix(size: int, indices: np.ndarray) -> np.ndarray:
    mat = np.zeros((indices.size, size), dtype=np.complex128)
    mat[np.arange(indices.size), indices] = 1.0
    return mat


def _draw_indices(
    rng: np.random.Generator,
    pool: np.ndarray,
    count: int,
    used: set[int],
    no_repeat: bool,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    if pool.size < count:
        raise ValueError("count cannot exceed pool size")

    if no_repeat:
        avail = [i for i in pool.tolist() if i not in used]
        if len(avail) >= count:
            chosen = np.asarray(rng.choice(avail, size=count, replace=False), dtype=int)
            used.update(chosen.tolist())
            return chosen

    chosen = np.asarray(rng.choice(pool, size=count, replace=False), dtype=int)
    if no_repeat:
        used.update(chosen.tolist())
    return chosen


def build_fpws_selection_contexts(
    n_rx_modes: int,
    n_tx_modes: int,
    n_slots: int,
    n_rx_rf: int,
    n_tx_rf: int,
    seed: int = 0,
    no_repeat: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """Build per-slot selection probing contexts on propagating modal indices."""

    if min(n_rx_modes, n_tx_modes, n_slots, n_rx_rf, n_tx_rf) <= 0:
        raise ValueError("all size parameters must be positive")

    rng = np.random.default_rng(seed)
    rx_pool = np.arange(n_rx_modes, dtype=int)
    tx_pool = np.arange(n_tx_modes, dtype=int)

    used_rx: set[int] = set()
    used_tx: set[int] = set()

    contexts: List[Dict[str, np.ndarray]] = []
    for _ in range(n_slots):
        rx_idx = _draw_indices(rng, rx_pool, n_rx_rf, used_rx, no_repeat)
        tx_idx = _draw_indices(rng, tx_pool, n_tx_rf, used_tx, no_repeat)

        q = _selection_matrix(n_rx_modes, rx_idx)
        p = _selection_matrix(n_tx_modes, tx_idx).T  # (n_tx_modes, n_tx_rf)
        x = p @ np.ones((n_tx_rf,), dtype=np.complex128)

        contexts.append({"Q": q, "P": p, "x": x, "rx_indices": rx_idx, "tx_indices": tx_idx})

    return contexts
