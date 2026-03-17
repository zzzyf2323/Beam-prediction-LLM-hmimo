"""Smoke tests for operator-form Group-LASSO."""

import numpy as np

from hmimo.estimators.group_lasso import solve_group_lasso
from hmimo.physics.operators import adjoint_operator, forward_operator


def test_group_lasso_operator_smoke() -> None:
    """Run tiny deterministic end-to-end Group-LASSO solve."""

    rng = np.random.default_rng(11)

    nr_modes = 3
    ns_modes = 2
    shape = (nr_modes, ns_modes)

    contexts = [
        {
            "Q": rng.normal(size=(2, nr_modes)) + 1j * rng.normal(size=(2, nr_modes)),
            "x": rng.normal(size=(ns_modes,)) + 1j * rng.normal(size=(ns_modes,)),
        },
        {
            "Q": rng.normal(size=(1, nr_modes)) + 1j * rng.normal(size=(1, nr_modes)),
            "x": rng.normal(size=(ns_modes,)) + 1j * rng.normal(size=(ns_modes,)),
        },
        {
            "Q": rng.normal(size=(nr_modes,)) + 1j * rng.normal(size=(nr_modes,)),
            "x": rng.normal(size=(ns_modes,)) + 1j * rng.normal(size=(ns_modes,)),
        },
    ]

    def fwd(ha: np.ndarray) -> np.ndarray:
        return forward_operator(ha, contexts)

    def adj(r: np.ndarray) -> np.ndarray:
        return adjoint_operator(r, contexts, ha_shape=shape)

    ha_true = np.zeros(shape, dtype=np.complex128)
    ha_true[0, 0] = 1.2 - 0.7j
    ha_true[2, 1] = -0.9 + 0.1j

    y = fwd(ha_true)

    dim = nr_modes * ns_modes
    groups = [[i] for i in range(dim)]

    result = solve_group_lasso(
        forward_op=fwd,
        adjoint_op=adj,
        y=y,
        shape=shape,
        groups=groups,
        lam=1e-2,
        max_iter=80,
        tol=1e-8,
    )

    ha_hat = result["ha_hat"]
    assert ha_hat.shape == shape

    obj_hist = np.asarray(result["objective_history"], dtype=float)
    assert obj_hist.size >= 1
    assert np.all(np.isfinite(obj_hist))
    assert obj_hist[-1] <= obj_hist[0] * 1.2 + 1e-12

    rel_err = np.linalg.norm(ha_hat - ha_true) / max(np.linalg.norm(ha_true), 1e-12)
    assert np.isfinite(rel_err)
