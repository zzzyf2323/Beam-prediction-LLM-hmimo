"""Evaluation metrics for HMIMO experiments."""

from __future__ import annotations

import numpy as np


def nmse_ha(ha_hat: np.ndarray, ha_true: np.ndarray, eps: float = 1e-12) -> float:
    """Compute NMSE for modal-domain channel estimate."""

    num = float(np.linalg.norm(ha_hat - ha_true) ** 2)
    den = float(np.linalg.norm(ha_true) ** 2)
    return num / max(den, eps)


def nmse_h(h_hat: np.ndarray, h_true: np.ndarray, eps: float = 1e-12) -> float:
    """Compute NMSE for spatial-domain channel estimate."""

    num = float(np.linalg.norm(h_hat - h_true) ** 2)
    den = float(np.linalg.norm(h_true) ** 2)
    return num / max(den, eps)
