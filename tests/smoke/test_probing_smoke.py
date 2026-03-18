"""Smoke tests for probing context builders."""

import numpy as np

from hmimo.physics.fpws import propagating_modes_rectangular
from hmimo.physics.operators import forward_operator
from hmimo.probing.dft_baseline import build_dft_baseline_contexts
from hmimo.probing.fpws_selection import build_fpws_selection_contexts


def test_probing_contexts_shapes_ranges_and_consumption() -> None:
    rx = propagating_modes_rectangular(2, 2)
    tx = propagating_modes_rectangular(2, 2)
    nr = rx.pairs.shape[0]
    ns = tx.pairs.shape[0]

    fpws = build_fpws_selection_contexts(
        n_rx_modes=nr,
        n_tx_modes=ns,
        n_slots=2,
        n_rx_rf=1,
        n_tx_rf=1,
        seed=3,
        no_repeat=True,
    )
    assert len(fpws) == 2
    rx_seen = []
    tx_seen = []
    for c in fpws:
        assert c["Q"].shape == (1, nr)
        assert c["P"].shape == (ns, 1)
        assert c["x"].shape == (ns,)
        assert np.all(c["rx_indices"] >= 0) and np.all(c["rx_indices"] < nr)
        assert np.all(c["tx_indices"] >= 0) and np.all(c["tx_indices"] < ns)
        rx_seen.extend(c["rx_indices"].tolist())
        tx_seen.extend(c["tx_indices"].tolist())
    assert len(set(rx_seen)) == len(rx_seen)
    assert len(set(tx_seen)) == len(tx_seen)

    dft = build_dft_baseline_contexts(
        rx_dims=(2, 2),
        tx_dims=(2, 2),
        rx_mode_pairs=rx.pairs,
        tx_mode_pairs=tx.pairs,
        n_slots=2,
        n_rx_rf=1,
        n_tx_rf=1,
        seed=4,
        no_repeat=True,
    )
    assert len(dft) == 2

    ha = np.ones((nr, ns), dtype=np.complex128)
    y_fpws = forward_operator(ha, fpws)
    y_dft = forward_operator(ha, dft)
    assert y_fpws.shape == (2,)
    assert y_dft.shape == (2,)
    assert np.all(np.isfinite(y_fpws))
    assert np.all(np.isfinite(y_dft))
