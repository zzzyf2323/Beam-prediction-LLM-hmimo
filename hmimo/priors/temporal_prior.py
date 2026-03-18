"""Non-LLM temporal warm-start baselines for dynamic HMIMO recovery."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def temporal_warm_start_from_prev(
    prev_ha_hat: Optional[np.ndarray],
    groups: list[list[int]],
    alpha: float = 0.7,
    prev_gamma: Optional[np.ndarray] = None,
    eps: float = 1e-10,
) -> Dict[str, np.ndarray | float]:
    """Build warm-start statistics from previous-frame estimate.

    Returns grouped energy estimate for SBL and a lambda scaling hint for
    Group-LASSO based on support persistence.
    """

    if prev_ha_hat is None:
        g = np.ones((len(groups),), dtype=float)
        return {"gamma_group_init": g, "lasso_lam_scale": 1.0}

    x = np.asarray(prev_ha_hat, dtype=np.complex128).reshape(-1)
    gamma = np.zeros((len(groups),), dtype=float)
    for i, grp in enumerate(groups):
        gi = np.asarray(grp, dtype=int)
        energy = float(np.vdot(x[gi], x[gi]).real) / max(float(gi.size), 1.0)
        gamma[i] = max(energy, eps)

    if prev_gamma is not None:
        pg = np.asarray(prev_gamma, dtype=float).reshape(-1)
        if pg.shape == gamma.shape:
            gamma = alpha * pg + (1.0 - alpha) * gamma

    strong_ratio = float(np.mean(gamma > np.median(gamma)))
    lasso_scale = float(np.clip(1.1 - 0.4 * strong_ratio, 0.7, 1.1))

    return {"gamma_group_init": gamma, "lasso_lam_scale": lasso_scale}
