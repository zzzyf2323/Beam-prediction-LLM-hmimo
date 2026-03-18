"""Tiny static comparison smoke test across four HMIMO core schemes."""

import numpy as np

from hmimo.estimators.group_lasso import solve_group_lasso
from hmimo.estimators.group_sbl import solve_group_sbl
from hmimo.grouping.groups import build_rx_tx_groups
from hmimo.physics.fpws import propagating_modes_rectangular
from hmimo.physics.operators import adjoint_operator, forward_operator
from hmimo.probing.dft_baseline import build_dft_baseline_contexts
from hmimo.probing.fpws_selection import build_fpws_selection_contexts


def _nmse(est: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(est - ref) ** 2 / max(np.linalg.norm(ref) ** 2, 1e-12))


def test_static_compare_four_schemes_smoke() -> None:
    rng = np.random.default_rng(19)

    rx = propagating_modes_rectangular(2, 2)
    tx = propagating_modes_rectangular(2, 2)
    nr, ns = rx.pairs.shape[0], tx.pairs.shape[0]
    shape = (nr, ns)

    grouping = build_rx_tx_groups(rx.pairs, tx.pairs, rx_block_shape=(1, 1), tx_block_shape=(1, 1))
    groups = [g.vector_indices.tolist() for g in grouping["groups"]]

    ha_true = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    fpws_ctx = build_fpws_selection_contexts(nr, ns, n_slots=4, n_rx_rf=1, n_tx_rf=1, seed=10, no_repeat=True)
    dft_ctx = build_dft_baseline_contexts(
        rx_dims=(2, 2),
        tx_dims=(2, 2),
        rx_mode_pairs=rx.pairs,
        tx_mode_pairs=tx.pairs,
        n_slots=4,
        n_rx_rf=1,
        n_tx_rf=1,
        seed=10,
        no_repeat=True,
    )

    def make_ops(contexts):
        def fwd(h):
            return forward_operator(h, contexts)

        def adj(r):
            return adjoint_operator(r, contexts, ha_shape=shape)

        return fwd, adj

    schemes = []
    for contexts, est_name in [(fpws_ctx, "lasso"), (fpws_ctx, "sbl"), (dft_ctx, "lasso"), (dft_ctx, "sbl")]:
        fwd, adj = make_ops(contexts)
        y = fwd(ha_true)
        if est_name == "lasso":
            out = solve_group_lasso(fwd, adj, y, shape, groups, lam=1e-2, max_iter=60, tol=1e-7)
        else:
            out = solve_group_sbl(fwd, adj, y, shape, groups, noise_var=1e-2, max_iter=25, tol=1e-6)
        ha_hat = out["ha_hat"]
        schemes.append((ha_hat, _nmse(ha_hat, ha_true)))

    for ha_hat, nmse in schemes:
        assert ha_hat.shape == shape
        assert np.all(np.isfinite(ha_hat))
        assert np.isfinite(nmse)
