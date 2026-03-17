"""Operator-form HMIMO modal sensing maps.

The forward and adjoint operators are implemented without constructing a giant
explicit sensing matrix in the normal path.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _as_complex_vector(x: np.ndarray, name: str) -> np.ndarray:
    v = np.asarray(x)
    if v.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    return v.astype(np.complex128, copy=False)


def _as_complex_matrix(x: np.ndarray, name: str) -> np.ndarray:
    m = np.asarray(x)
    if m.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix")
    return m.astype(np.complex128, copy=False)


def _normalize_context(context: dict, nr_modes: int, ns_modes: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Validate a sensing context and return normalized ``(Q, x, out_dim)``."""

    if "x" not in context or "Q" not in context:
        raise ValueError("Each context must include keys 'Q' and 'x'")

    x = _as_complex_vector(context["x"], "x")
    if x.shape[0] != ns_modes:
        raise ValueError("x has incompatible length with Ha transmit dimension")

    q_raw = np.asarray(context["Q"])
    if q_raw.ndim == 1:
        q = _as_complex_vector(q_raw, "Q")
        if q.shape[0] != nr_modes:
            raise ValueError("Q vector has incompatible length with Ha receive dimension")
        return q, x, 1

    q = _as_complex_matrix(q_raw, "Q")
    if q.shape[1] != nr_modes:
        raise ValueError("Q matrix has incompatible width with Ha receive dimension")
    return q, x, q.shape[0]


def forward_operator(ha: np.ndarray, contexts: Sequence[dict]) -> np.ndarray:
    """Apply modal sensing forward operator ``A(ha, contexts)``.

    Each context provides:
    * ``x``: transmit probe vector with shape ``(N_s,)``.
    * ``Q``: receive selector/combiner as shape ``(N_r,)`` or ``(M_p, N_r)``.

    Returns:
        Concatenated complex residual vector across all contexts.
    """

    ha_mat = _as_complex_matrix(ha, "ha")
    nr_modes, ns_modes = ha_mat.shape

    outputs: List[np.ndarray] = []
    for context in contexts:
        q, x, out_dim = _normalize_context(context, nr_modes, ns_modes)
        hx = ha_mat @ x
        if out_dim == 1:
            outputs.append(np.asarray(q @ hx, dtype=np.complex128).reshape(-1))
        else:
            outputs.append(q @ hx)

    if not outputs:
        return np.zeros((0,), dtype=np.complex128)
    return np.concatenate(outputs, axis=0)


def adjoint_operator(r: np.ndarray, contexts: Sequence[dict], ha_shape: Tuple[int, int]) -> np.ndarray:
    """Apply modal sensing adjoint map ``A^H(r, contexts)``.

    Args:
        r: Stacked residual vector (same layout as ``forward_operator`` output).
        contexts: Sequence of sensing contexts.
        ha_shape: Target output matrix shape ``(N_r, N_s)``.
    """

    nr_modes, ns_modes = ha_shape
    residual = _as_complex_vector(r, "r")
    out = np.zeros(ha_shape, dtype=np.complex128)

    offset = 0
    for context in contexts:
        q, x, out_dim = _normalize_context(context, nr_modes, ns_modes)
        chunk = residual[offset : offset + out_dim]
        if chunk.shape[0] != out_dim:
            raise ValueError("Residual vector length does not match contexts")

        if out_dim == 1:
            z = q.conj() * chunk[0]
        else:
            z = q.conj().T @ chunk

        out += np.outer(z, x.conj())
        offset += out_dim

    if offset != residual.shape[0]:
        raise ValueError("Residual vector has extra entries not used by contexts")

    return out
