"""Smoke tests for stage-1 FPWS dictionaries and operators."""

import numpy as np

from hmimo.physics.fpws import fpws_dictionaries_rectangular, propagating_modes_rectangular
from hmimo.physics.operators import adjoint_operator, forward_operator


def test_fpws_and_operator_adjoint_smoke() -> None:
    """Validate tiny-shape behavior and adjoint consistency."""

    rng = np.random.default_rng(7)

    psi_r, psi_s = fpws_dictionaries_rectangular((2, 2), (2, 1))
    assert psi_r.shape == (4, 4)
    assert psi_s.shape == (2, 2)

    modes = propagating_modes_rectangular(nx=2, ny=2, d_over_lambda=0.5)
    assert modes.pairs.ndim == 2
    assert modes.pairs.shape[1] == 2

    nr_modes = 4
    ns_modes = 3
    ha = rng.normal(size=(nr_modes, ns_modes)) + 1j * rng.normal(size=(nr_modes, ns_modes))

    contexts = [
        {
            "Q": rng.normal(size=(2, nr_modes)) + 1j * rng.normal(size=(2, nr_modes)),
            "x": rng.normal(size=(ns_modes,)) + 1j * rng.normal(size=(ns_modes,)),
        },
        {
            "Q": rng.normal(size=(nr_modes,)) + 1j * rng.normal(size=(nr_modes,)),
            "x": rng.normal(size=(ns_modes,)) + 1j * rng.normal(size=(ns_modes,)),
        },
    ]

    ah = forward_operator(ha, contexts)
    assert ah.shape == (3,)

    r = rng.normal(size=ah.shape) + 1j * rng.normal(size=ah.shape)
    lhs = np.vdot(ah, r)
    rhs = np.vdot(ha, adjoint_operator(r, contexts, ha_shape=ha.shape))

    assert np.allclose(lhs, rhs, atol=1e-10, rtol=1e-10)
