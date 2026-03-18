"""Operator-form Group-SBL for HMIMO modal estimation.

This implementation uses grouped ARD hyperparameters and computes a posterior
mean through iterative linear solves with forward/adjoint callbacks, avoiding
explicit giant sensing matrices in the normal path.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


Array = np.ndarray


def _flatten(ha: Array) -> Array:
    """Flatten ``Ha`` matrix to vector in C-order."""

    return np.asarray(ha, dtype=np.complex128).reshape(-1)


def _reshape(v: Array, shape: Tuple[int, int]) -> Array:
    """Reshape vector to ``Ha`` matrix in C-order."""

    return np.asarray(v, dtype=np.complex128).reshape(shape)


def _validate_groups(groups: Sequence[Sequence[int]], dim: int) -> List[np.ndarray]:
    """Validate group index sets."""

    out: List[np.ndarray] = []
    for i, g in enumerate(groups):
        gi = np.asarray(g, dtype=int)
        if gi.ndim != 1 or gi.size == 0:
            raise ValueError(f"Group {i} must be non-empty and 1-D")
        if np.any(gi < 0) or np.any(gi >= dim):
            raise ValueError(f"Group {i} has out-of-range entries")
        out.append(gi)
    return out


def _expand_gamma(group_gamma: Array, groups: Sequence[np.ndarray], dim: int, eps: float) -> Array:
    """Expand grouped gamma values to per-coefficient variances."""

    gamma_diag = np.full((dim,), eps, dtype=float)
    for gid, g in enumerate(groups):
        gamma_diag[g] = max(float(group_gamma[gid]), eps)
    return gamma_diag


def _cg_solve(
    matvec: Callable[[Array], Array],
    b: Array,
    tol: float = 1e-8,
    max_iter: int = 300,
) -> Array:
    """Conjugate-gradient solve for Hermitian positive-definite systems."""

    x = np.zeros_like(b, dtype=np.complex128)
    r = b - matvec(x)
    p = r.copy()
    rs_old = np.vdot(r, r).real

    if rs_old <= 0.0:
        return x

    for _ in range(max_iter):
        ap = matvec(p)
        denom = np.vdot(p, ap).real
        if denom <= 0.0:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = np.vdot(r, r).real
        if np.sqrt(rs_new) <= tol:
            break
        beta = rs_new / max(rs_old, 1e-30)
        p = r + beta * p
        rs_old = rs_new

    return x


def solve_group_sbl(
    forward_op: Callable[[Array], Array],
    adjoint_op: Callable[[Array], Array],
    y: Array,
    shape: Tuple[int, int],
    groups: Sequence[Sequence[int]],
    noise_var: float,
    max_iter: int = 50,
    tol: float = 1e-5,
    damping: float = 0.4,
    gamma_init: float = 1.0,
    eps: float = 1e-10,
    cg_tol: float = 1e-8,
    cg_max_iter: int = 300,
) -> Dict[str, object]:
    """Solve grouped SBL using operator-form posterior mean updates.

    The posterior mean ``x`` (vectorized ``Ha``) is approximated by solving
    ``(A^H A + noise_var * Gamma^{-1}) x = A^H y`` with CG.
    Group variances are then updated from group-wise posterior mean energy.
    """

    if noise_var <= 0:
        raise ValueError("noise_var must be positive")
    if not (0.0 <= damping < 1.0):
        raise ValueError("damping must be in [0, 1)")

    y_vec = np.asarray(y, dtype=np.complex128).reshape(-1)
    dim = int(np.prod(shape))
    group_idx = _validate_groups(groups, dim)

    gamma_group = np.full((len(group_idx),), float(gamma_init), dtype=float)
    gamma_history: List[np.ndarray] = []

    rhs = _flatten(adjoint_op(y_vec))
    x = np.zeros((dim,), dtype=np.complex128)

    converged = False
    for _ in range(max_iter):
        gamma_diag = _expand_gamma(gamma_group, group_idx, dim, eps)
        inv_gamma = 1.0 / gamma_diag

        def system_matvec(v: Array) -> Array:
            va = _reshape(v, shape)
            ah_a_v = _flatten(adjoint_op(forward_op(va)))
            return ah_a_v + noise_var * inv_gamma * v

        x = _cg_solve(system_matvec, rhs, tol=cg_tol, max_iter=cg_max_iter)

        gamma_new = np.zeros_like(gamma_group)
        for gid, g in enumerate(group_idx):
            group_energy = float(np.vdot(x[g], x[g]).real)
            gamma_new[gid] = max(group_energy / float(g.size), eps)

        gamma_new = damping * gamma_group + (1.0 - damping) * gamma_new
        gamma_history.append(gamma_new.copy())

        rel = np.linalg.norm(gamma_new - gamma_group) / max(np.linalg.norm(gamma_group), 1.0)
        gamma_group = gamma_new
        if rel < tol:
            converged = True
            break

    ha_hat = _reshape(x, shape)
    return {
        "ha_hat": ha_hat,
        "gamma_group": gamma_group,
        "gamma_history": gamma_history,
        "converged": converged,
    }
