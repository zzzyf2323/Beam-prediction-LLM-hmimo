"""Minimal LLM-assisted temporal prior for HMIMO dynamic recovery.

The module only produces prior hints (gamma initialization and group activity
probabilities). It does not estimate channels directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LLMTemporalPriorConfig:
    """Configuration for lightweight LLM-assisted prior inference."""

    enabled: bool = True
    history_len: int = 3
    fusion_dim: int = 32
    seed: int = 0


class HMIMOLLMTemporalPrior:
    """Lightweight prior module with LLM-style feature fusion over short history."""

    def __init__(self, config: LLMTemporalPriorConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self._proj = rng.normal(scale=0.2, size=(6, config.fusion_dim))
        self._head_gamma = rng.normal(scale=0.2, size=(config.fusion_dim,))
        self._head_prob = rng.normal(scale=0.2, size=(config.fusion_dim,))

    def _frame_features(
        self,
        frame: Dict[str, np.ndarray | float],
        meta: Dict[str, float],
        use_metadata: bool,
        use_gamma: bool,
        use_energy: bool,
        use_activity: bool,
    ) -> np.ndarray:
        g = np.asarray(frame.get("gamma", 0.0), dtype=float) if use_gamma else np.zeros((1,), dtype=float)
        e = np.asarray(frame.get("energy", 0.0), dtype=float) if use_energy else np.zeros((1,), dtype=float)
        p = np.asarray(frame.get("active_prob", 0.0), dtype=float) if use_activity else np.zeros((1,), dtype=float)

        meta_groups = float(meta["n_groups"]) if use_metadata else 0.0
        meta_rf = float(meta["rf_budget"]) if use_metadata else 0.0

        return np.array(
            [
                float(np.mean(g)),
                float(np.mean(e)),
                float(np.mean(p)),
                float(frame["snr_db"]),
                meta_groups,
                meta_rf,
            ],
            dtype=float,
        )

    def infer_prior(
        self,
        history: List[Dict[str, np.ndarray | float]],
        metadata: Dict[str, float],
        fallback_gamma: np.ndarray,
        ablation: Optional[Dict[str, object]] = None,
    ) -> Dict[str, np.ndarray]:
        """Infer prior hints from compact summaries and metadata conditioning."""

        ablation = ablation or {}
        use_metadata = bool(ablation.get("use_metadata", True))
        use_gamma = bool(ablation.get("use_gamma", True))
        use_energy = bool(ablation.get("use_energy", True))
        use_activity = bool(ablation.get("use_activity", True))
        history_len = int(ablation.get("history_len", self.config.history_len))

        base_gamma = np.asarray(fallback_gamma, dtype=float)
        n_groups = base_gamma.size

        if (not self.config.enabled) or len(history) == 0:
            prob = np.clip(base_gamma / (np.max(base_gamma) + 1e-12), 0.0, 1.0)
            return {"gamma_init": np.maximum(base_gamma, 1e-10), "activity_prob": prob}

        hist = history[-history_len:]
        feats = np.stack(
            [self._frame_features(h, metadata, use_metadata, use_gamma, use_energy, use_activity) for h in hist],
            axis=0,
        )

        fused = np.tanh(feats @ self._proj)
        token = np.mean(fused, axis=0)

        gamma_scale = 1.0 + 0.25 * float(np.tanh(token @ self._head_gamma))
        prob_shift = 0.25 * float(np.tanh(token @ self._head_prob))

        last_energy = np.asarray(hist[-1].get("energy", base_gamma), dtype=float)
        if not use_energy:
            last_energy = np.asarray(base_gamma, dtype=float)
        gamma_init = np.maximum(0.6 * base_gamma + 0.4 * gamma_scale * last_energy, 1e-10)

        prob = last_energy / (np.max(last_energy) + 1e-12)
        prob = np.clip(prob + prob_shift, 0.0, 1.0)

        if prob.size != n_groups:
            prob = np.resize(prob, n_groups)
        if gamma_init.size != n_groups:
            gamma_init = np.resize(gamma_init, n_groups)

        return {"gamma_init": gamma_init, "activity_prob": prob}
