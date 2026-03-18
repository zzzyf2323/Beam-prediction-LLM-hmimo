"""Smoke tests for operator-form Group-SBL."""

import numpy as np

from hmimo.estimators.group_sbl import solve_group_sbl
from hmimo.grouping.groups import build_rx_tx_groups
from hmimo.physics.operators import adjoint_operator, forward_operator


def test_group_sbl_operator_smoke() -> None:
    """Run tiny deterministic end-to-end Group-SBL solve."""

    rng = np.random.default_rng(23)

    nr_modes = 3
    ns_modes = 3
    shape = (nr_modes, ns_modes)

    rx_modes = np.array([[-1, 0], [0, 0], [1, 0]])
    tx_modes = np.array([[-1, 0], [0, 0], [1, 0]])
    grouping = build_rx_tx_groups(
        rx_mode_pairs=rx_modes,
        tx_mode_pairs=tx_modes,
        rx_block_shape=(1, 1),
        tx_block_shape=(1, 1),
    )
    groups = [g.vector_indices.tolist() for g in grouping["groups"]]

    contexts = [
        {
            "Q": rng.normal(size=(2, nr_modes)) + 1j * rng.normal(size=(2, nr_modes)),
            "x": rng.normal(size=(ns_modes,)) + 1j * rng.normal(size=(ns_modes,)),
        },
        {
            "Q": rng.normal(size=(nr_modes,)) + 1j * rng.normal(size=(nr_modes,)),
            "x": rng.normal(size=(ns_modes,)) + 1j * rng.normal(size=(ns_modes,)),
        },
        {
            "Q": rng.normal(size=(1, nr_modes)) + 1j * rng.normal(size=(1, nr_modes)),
            "x": rng.normal(size=(ns_modes,)) + 1j * rng.normal(size=(ns_modes,)),
        },
    ]

    def fwd(ha: np.ndarray) -> np.ndarray:
        return forward_operator(ha, contexts)

    def adj(r: np.ndarray) -> np.ndarray:
        return adjoint_operator(r, contexts, ha_shape=shape)

    ha_true = np.zeros(shape, dtype=np.complex128)
    ha_true[0, 0] = 1.1 - 0.4j
    ha_true[2, 2] = -0.8 + 0.2j

    y = fwd(ha_true)

    out = solve_group_sbl(
        forward_op=fwd,
        adjoint_op=adj,
        y=y,
        shape=shape,
        groups=groups,
        noise_var=1e-2,
        max_iter=30,
        tol=1e-6,
        damping=0.4,
    )

    ha_hat = out["ha_hat"]
    gamma = np.asarray(out["gamma_group"], dtype=float)

    assert ha_hat.shape == shape
    assert gamma.shape[0] == len(groups)
    assert np.all(np.isfinite(gamma))
    assert np.all(gamma >= 0.0)

    rel_err = np.linalg.norm(ha_hat - ha_true) / max(np.linalg.norm(ha_true), 1e-12)
    assert np.isfinite(rel_err)
