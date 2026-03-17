"""Grouping utilities for HMIMO modal coefficient structure.

This module builds deterministic local neighborhoods on receive/transmit
propagating-mode lattices and then forms Cartesian-product matrix groups.
The resulting vectorized indices follow C-order ``vec(Ha)`` convention used by
estimators in this repository: ``idx = rx_idx * n_tx_modes + tx_idx``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ModeNeighborhood:
    """Localized neighborhood over modal indices for one side (RX or TX)."""

    block_id: int
    block_key: Tuple[int, int]
    member_indices: np.ndarray
    member_modes: np.ndarray


@dataclass(frozen=True)
class MatrixGroup:
    """Cartesian-product group on matrix/vectorized modal coefficients."""

    group_id: int
    rx_block_id: int
    tx_block_id: int
    rx_member_indices: np.ndarray
    tx_member_indices: np.ndarray
    vector_indices: np.ndarray


def _validate_mode_pairs(mode_pairs: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(mode_pairs, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (K, 2)")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def partition_mode_neighborhoods(
    mode_pairs: np.ndarray,
    block_shape: Tuple[int, int],
) -> List[ModeNeighborhood]:
    """Partition propagating modes into deterministic local neighborhoods.

    Args:
        mode_pairs: Integer modal pairs ``(m_x, m_y)`` with shape ``(K, 2)``.
            The row order defines the estimator mode index order.
        block_shape: Positive block shape ``(bx, by)`` on the mode lattice.

    Returns:
        List of neighborhoods with deterministic ordering by lattice block key.
    """

    pairs = _validate_mode_pairs(mode_pairs, "mode_pairs")
    bx, by = block_shape
    if bx <= 0 or by <= 0:
        raise ValueError("block_shape entries must be positive")

    min_x = int(np.min(pairs[:, 0]))
    min_y = int(np.min(pairs[:, 1]))

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for idx, (mx, my) in enumerate(pairs):
        key = ((int(mx) - min_x) // int(bx), (int(my) - min_y) // int(by))
        buckets.setdefault(key, []).append(idx)

    neighborhoods: List[ModeNeighborhood] = []
    for block_id, key in enumerate(sorted(buckets.keys())):
        member_indices = np.asarray(sorted(buckets[key]), dtype=int)
        neighborhoods.append(
            ModeNeighborhood(
                block_id=block_id,
                block_key=key,
                member_indices=member_indices,
                member_modes=pairs[member_indices],
            )
        )

    return neighborhoods


def build_cartesian_matrix_groups(
    rx_neighborhoods: Sequence[ModeNeighborhood],
    tx_neighborhoods: Sequence[ModeNeighborhood],
    n_tx_modes: int,
) -> List[MatrixGroup]:
    """Build matrix/vector groups ``G_{u,v} = B_u^(R) x B_v^(S)``.

    Args:
        rx_neighborhoods: Receive neighborhoods partitioning RX mode indices.
        tx_neighborhoods: Transmit neighborhoods partitioning TX mode indices.
        n_tx_modes: Total number of TX modes (matrix width).

    Returns:
        Deterministically ordered matrix groups over vectorized ``Ha`` indices.
    """

    if n_tx_modes <= 0:
        raise ValueError("n_tx_modes must be positive")

    groups: List[MatrixGroup] = []
    gid = 0
    for rx_block in rx_neighborhoods:
        for tx_block in tx_neighborhoods:
            vec_idx = [
                int(r) * int(n_tx_modes) + int(c)
                for r in rx_block.member_indices
                for c in tx_block.member_indices
            ]
            groups.append(
                MatrixGroup(
                    group_id=gid,
                    rx_block_id=rx_block.block_id,
                    tx_block_id=tx_block.block_id,
                    rx_member_indices=np.asarray(rx_block.member_indices, dtype=int),
                    tx_member_indices=np.asarray(tx_block.member_indices, dtype=int),
                    vector_indices=np.asarray(vec_idx, dtype=int),
                )
            )
            gid += 1

    return groups


def build_rx_tx_groups(
    rx_mode_pairs: np.ndarray,
    tx_mode_pairs: np.ndarray,
    rx_block_shape: Tuple[int, int],
    tx_block_shape: Tuple[int, int],
) -> Dict[str, object]:
    """Convenience constructor for RX/TX neighborhoods and matrix groups."""

    rx_pairs = _validate_mode_pairs(rx_mode_pairs, "rx_mode_pairs")
    tx_pairs = _validate_mode_pairs(tx_mode_pairs, "tx_mode_pairs")

    rx_neighborhoods = partition_mode_neighborhoods(rx_pairs, rx_block_shape)
    tx_neighborhoods = partition_mode_neighborhoods(tx_pairs, tx_block_shape)
    groups = build_cartesian_matrix_groups(
        rx_neighborhoods=rx_neighborhoods,
        tx_neighborhoods=tx_neighborhoods,
        n_tx_modes=tx_pairs.shape[0],
    )

    return {
        "rx_neighborhoods": rx_neighborhoods,
        "tx_neighborhoods": tx_neighborhoods,
        "groups": groups,
    }
