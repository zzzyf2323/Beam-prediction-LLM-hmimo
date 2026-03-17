"""Operator-form Group-LASSO solver for HMIMO modal estimation.

This module implements a lightweight FISTA routine that works with forward and
adjoint operator callbacks. It avoids constructing explicit large sensing
matrices in the normal code path.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


def _flatten(ha: Array) -> Array:
    """Flatten channel matrix to a 1-D vector in C order."""

    return np.asarray(ha, dtype=np.complex128).reshape(-1)


def _reshape(v: Array, shape: Tuple[int, int]) -> Array:
    """Reshape a 1-D vector to channel matrix shape in C order."""

    return np.asarray(v, dtype=np.complex128).reshape(shape)


def _validate_groups(groups: Sequence[Sequence[int]], dim: int) -> List[np.ndarray]:
    """Validate and normalize group indices."""

    normalized: List[np.ndarray] = []
    for idx, group in enumerate(groups):
        g = np.asarray(group, dtype=int)
        if g.ndim != 1 or g.size == 0:
            raise ValueError(f"Group {idx} must be a non-empty 1-D index list")
        if np.any(g < 0) or np.any(g >= dim):
            raise ValueError(f"Group {idx} has out-of-range indices")
        normalized.append(g)
    return normalized


def _group_soft_threshold(v: Array, tau: float, groups: Sequence[np.ndarray]) -> Array:
    """Apply group-wise l2 soft-thresholding prox."""

    out = np.asarray(v, dtype=np.complex128).copy()
    for g in groups:
        block = out[g]
        norm = np.linalg.norm(block)
        if norm <= tau:
            out[g] = 0.0
        else:
            out[g] = (1.0 - tau / norm) * block
    return out


def _objective(
    x: Array,
    shape: Tuple[int, int],
    y: Array,
    forward_op: Callable[[Array], Array],
    groups: Sequence[np.ndarray],
    lam: float,
) -> float:
    """Evaluate Group-LASSO objective value."""

    residual = forward_op(_reshape(x, shape)) - y
    data_term = 0.5 * float(np.vdot(residual, residual).real)
    reg = lam * sum(float(np.linalg.norm(x[g])) for g in groups)
    return data_term + reg


def _estimate_step_size(
    forward_op: Callable[[Array], Array],
    adjoint_op: Callable[[Array], Array],
    shape: Tuple[int, int],
    seed: int = 0,
    num_power_iter: int = 30,
) -> float:
    """Estimate FISTA step size as reciprocal Lipschitz constant."""

    rng = np.random.default_rng(seed)
    z = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    z = z.astype(np.complex128)
    z /= np.linalg.norm(z)

    eig_est = 1.0
    for _ in range(num_power_iter):
        z = adjoint_op(forward_op(z))
        nz = np.linalg.norm(z)
        if nz == 0.0:
            return 1.0
        z /= nz
        eig_est = float(np.vdot(z, adjoint_op(forward_op(z))).real)

    lipschitz = max(eig_est, 1e-12)
    return 1.0 / lipschitz


def solve_group_lasso(
    forward_op: Callable[[Array], Array],
    adjoint_op: Callable[[Array], Array],
    y: Array,
    shape: Tuple[int, int],
    groups: Sequence[Sequence[int]],
    lam: float,
    max_iter: int = 200,
    tol: float = 1e-6,
    step_size: Optional[float] = None,
) -> Dict[str, Array | float | int | List[float]]:
    """Solve Group-LASSO with operator-form FISTA.

    Minimizes ``0.5||A(Ha)-y||_2^2 + lam * sum_g ||vec(Ha)_g||_2``.
    """

    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    dim = int(np.prod(shape))
    group_idx = _validate_groups(groups, dim)
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0:
        raise ValueError("tol must be non-negative")

    if step_size is None:
        step_size = _estimate_step_size(forward_op, adjoint_op, shape)
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    x = np.zeros((dim,), dtype=np.complex128)
    z = x.copy()
    t = 1.0

    objective_history: List[float] = []
    converged = False

    for iteration in range(1, max_iter + 1):
        z_mat = _reshape(z, shape)
        grad_mat = adjoint_op(forward_op(z_mat) - y)
        grad = _flatten(grad_mat)

        x_next = _group_soft_threshold(z - step_size * grad, step_size * lam, group_idx)

        obj = _objective(x_next, shape, y, forward_op, group_idx, lam)
        objective_history.append(obj)

        rel_change = np.linalg.norm(x_next - x) / max(np.linalg.norm(x), 1.0)
        if rel_change < tol:
            x = x_next
            converged = True
            break

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_next + ((t - 1.0) / t_next) * (x_next - x)

        x = x_next
        t = t_next

    ha_hat = _reshape(x, shape)
    return {
        "ha_hat": ha_hat,
        "objective_history": objective_history,
        "iterations": iteration,
        "converged": converged,
        "step_size": float(step_size),
    }
