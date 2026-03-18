"""Smoke tests for HMIMO grouping utilities."""

import numpy as np

from hmimo.grouping.groups import build_rx_tx_groups


def test_group_partition_and_vec_indexing_smoke() -> None:
    """Verify deterministic partitioning and vec(Ha) index consistency."""

    # Deterministic tiny modal lattices; row order is the estimator index order.
    rx_modes = np.array([
        [-1, 0],
        [0, 0],
        [1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
    ])
    tx_modes = np.array([
        [-1, 0],
        [0, 0],
        [-1, 1],
        [0, 1],
    ])

    out = build_rx_tx_groups(
        rx_mode_pairs=rx_modes,
        tx_mode_pairs=tx_modes,
        rx_block_shape=(2, 1),
        tx_block_shape=(1, 1),
    )

    groups = out["groups"]
    n_rx = rx_modes.shape[0]
    n_tx = tx_modes.shape[0]
    total = n_rx * n_tx

    # Coverage and no-overlap checks.
    all_idx = np.concatenate([g.vector_indices for g in groups], axis=0)
    assert all_idx.size == total
    assert np.unique(all_idx).size == total
    assert np.array_equal(np.sort(all_idx), np.arange(total))

    # In-range checks.
    assert np.all(all_idx >= 0)
    assert np.all(all_idx < total)

    # Vec(Ha) convention checks: idx = rx_idx * n_tx + tx_idx (C-order flatten).
    for g in groups:
        for r in g.rx_member_indices:
            for c in g.tx_member_indices:
                expected = int(r) * n_tx + int(c)
                assert expected in set(g.vector_indices.tolist())
