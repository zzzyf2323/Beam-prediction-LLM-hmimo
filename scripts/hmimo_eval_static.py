"""Minimal reproducible static HMIMO evaluation harness with Monte Carlo support."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hmimo.estimators.group_lasso import solve_group_lasso
from hmimo.estimators.group_sbl import solve_group_sbl
from hmimo.evaluation.metrics import nmse_h, nmse_ha
from hmimo.grouping.groups import build_rx_tx_groups
from hmimo.physics.operators import adjoint_operator, forward_operator
from hmimo.probing.dft_baseline import build_dft_baseline_contexts
from hmimo.probing.fpws_selection import build_fpws_selection_contexts
from hmimo.simulation.channel import build_propagating_fpws_bases, generate_static_hmimo_channel

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


def _parse_scalar(value: str):
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v.startswith("[") and v.endswith("]"):
        body = v[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(x.strip()) for x in body.split(",")]
    try:
        if any(ch in v for ch in [".", "e", "E"]):
            return float(v)
        return int(v)
    except ValueError:
        return v


def _load_config(path: Path):
    if yaml is not None:
        return yaml.safe_load(path.read_text())

    lines = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            lines.append(line)

    root = {}
    stack = [(-1, root)]
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        key, _, rest = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        rest = rest.strip()
        if rest == "":
            parent[key] = {}
            stack.append((indent, parent[key]))
        else:
            parent[key] = _parse_scalar(rest)
    return root


def _reconstruct_h(ha: np.ndarray, psi_r_prop: np.ndarray, psi_s_prop: np.ndarray) -> np.ndarray:
    return psi_r_prop @ ha @ psi_s_prop.conj().T


def _aggregate_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, float], List[Dict[str, object]]] = {}
    for rec in records:
        key = (str(rec["scheme"]), float(rec["snr_db"]))
        buckets.setdefault(key, []).append(rec)

    summary: List[Dict[str, object]] = []
    for (scheme, snr_db), rows in sorted(buckets.items(), key=lambda x: (x[0][1], x[0][0])):
        nmse_ha_vals = np.asarray([float(r["nmse_ha"]) for r in rows], dtype=float)
        nmse_h_vals = np.asarray([float(r["nmse_h"]) for r in rows], dtype=float)
        summary.append(
            {
                "scheme": scheme,
                "snr_db": snr_db,
                "n_mc": len(rows),
                "mean_nmse_ha": float(np.mean(nmse_ha_vals)),
                "std_nmse_ha": float(np.std(nmse_ha_vals)),
                "mean_nmse_h": float(np.mean(nmse_h_vals)),
                "std_nmse_h": float(np.std(nmse_h_vals)),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snr_list = [float(x) for x in cfg["snr_db_list"]]
    n_mc = int(cfg.get("n_mc", 1))
    base_seed = int(cfg["seed"])

    lasso_cfg = cfg["group_lasso"]
    sbl_cfg = cfg["group_sbl"]

    records: List[Dict[str, object]] = []

    for trial_idx in range(n_mc):
        trial_seed = base_seed + 1000 * trial_idx

        ch_cfg = dict(cfg["channel"])
        ch_cfg["seed"] = int(ch_cfg.get("seed", base_seed)) + trial_idx
        sim = generate_static_hmimo_channel(ch_cfg)

        ha_true = sim["ha"]
        h_true = sim["h"]
        rx_pairs = sim["rx_mode_pairs"]
        tx_pairs = sim["tx_mode_pairs"]

        rx_dims = tuple(cfg["channel"]["rx_dims"])
        tx_dims = tuple(cfg["channel"]["tx_dims"])
        psi_r_prop, psi_s_prop = build_propagating_fpws_bases(
            rx_dims=rx_dims,
            tx_dims=tx_dims,
            rx_mode_pairs=rx_pairs,
            tx_mode_pairs=tx_pairs,
        )

        grouping = build_rx_tx_groups(
            rx_mode_pairs=rx_pairs,
            tx_mode_pairs=tx_pairs,
            rx_block_shape=tuple(cfg["grouping"]["rx_block_shape"]),
            tx_block_shape=tuple(cfg["grouping"]["tx_block_shape"]),
        )
        groups = [g.vector_indices.tolist() for g in grouping["groups"]]

        probing = cfg["probing"]
        fpws_contexts = build_fpws_selection_contexts(
            n_rx_modes=rx_pairs.shape[0],
            n_tx_modes=tx_pairs.shape[0],
            n_slots=int(probing["n_slots"]),
            n_rx_rf=int(probing["n_rx_rf"]),
            n_tx_rf=int(probing["n_tx_rf"]),
            seed=trial_seed,
            no_repeat=bool(probing.get("no_repeat", True)),
        )
        dft_contexts = build_dft_baseline_contexts(
            rx_dims=rx_dims,
            tx_dims=tx_dims,
            rx_mode_pairs=rx_pairs,
            tx_mode_pairs=tx_pairs,
            n_slots=int(probing["n_slots"]),
            n_rx_rf=int(probing["n_rx_rf"]),
            n_tx_rf=int(probing["n_tx_rf"]),
            seed=trial_seed,
            no_repeat=bool(probing.get("no_repeat", True)),
        )

        # Prebuild ops and noiseless measurements: same channel in all 4 schemes in each trial.
        scheme_contexts = {
            "fpws+group_lasso": fpws_contexts,
            "fpws+group_sbl": fpws_contexts,
            "dft+group_lasso": dft_contexts,
            "dft+group_sbl": dft_contexts,
        }
        noiseless: Dict[str, np.ndarray] = {}
        ops: Dict[str, Tuple] = {}
        for scheme, contexts in scheme_contexts.items():
            fwd = lambda h, c=contexts: forward_operator(h, c)
            adj = lambda r, c=contexts: adjoint_operator(r, c, ha_shape=ha_true.shape)
            ops[scheme] = (fwd, adj)
            noiseless[scheme] = fwd(ha_true)

        for snr_db in snr_list:
            signal_power = float(np.mean(np.abs(ha_true) ** 2))
            noise_var = signal_power / (10.0 ** (snr_db / 10.0))

            # Same trial+SNR noise seed; same base noise pattern used across schemes.
            nrng = np.random.default_rng(trial_seed + int(round(10 * snr_db)) + 17)
            ref_len = noiseless["fpws+group_lasso"].shape[0]
            base_noise = np.sqrt(noise_var / 2.0) * (
                nrng.normal(size=(ref_len,)) + 1j * nrng.normal(size=(ref_len,))
            )

            for scheme, (fwd, adj) in ops.items():
                y_clean = noiseless[scheme]
                noise = base_noise
                if y_clean.shape[0] != base_noise.shape[0]:
                    # Safety path for any future shape mismatch.
                    nrng_local = np.random.default_rng(trial_seed + int(round(10 * snr_db)) + hash(scheme) % 97)
                    noise = np.sqrt(noise_var / 2.0) * (
                        nrng_local.normal(size=y_clean.shape) + 1j * nrng_local.normal(size=y_clean.shape)
                    )
                y = y_clean + noise

                if scheme.endswith("group_lasso"):
                    out = solve_group_lasso(
                        forward_op=fwd,
                        adjoint_op=adj,
                        y=y,
                        shape=ha_true.shape,
                        groups=groups,
                        lam=float(lasso_cfg["lam"]),
                        max_iter=int(lasso_cfg["max_iter"]),
                        tol=float(lasso_cfg["tol"]),
                    )
                else:
                    out = solve_group_sbl(
                        forward_op=fwd,
                        adjoint_op=adj,
                        y=y,
                        shape=ha_true.shape,
                        groups=groups,
                        noise_var=max(noise_var, 1e-10),
                        max_iter=int(sbl_cfg["max_iter"]),
                        tol=float(sbl_cfg["tol"]),
                        damping=float(sbl_cfg["damping"]),
                    )

                ha_hat = out["ha_hat"]
                h_hat = _reconstruct_h(ha_hat, psi_r_prop, psi_s_prop)
                records.append(
                    {
                        "trial": trial_idx,
                        "scheme": scheme,
                        "snr_db": snr_db,
                        "nmse_ha": nmse_ha(ha_hat, ha_true),
                        "nmse_h": nmse_h(h_hat, h_true),
                    }
                )

    summary = _aggregate_records(records)

    json_path = out_dir / "hmimo_static_results.json"
    csv_path = out_dir / "hmimo_static_summary.csv"

    json_path.write_text(json.dumps({"config": cfg, "trial_results": records, "summary": summary}, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scheme", "snr_db", "n_mc", "mean_nmse_ha", "std_nmse_ha", "mean_nmse_h", "std_nmse_h"],
        )
        writer.writeheader()
        writer.writerows(summary)

    print("Static HMIMO Monte Carlo summary")
    print("scheme              | snr_db | n_mc | mean_nmse_ha | std_nmse_ha | mean_nmse_h | std_nmse_h")
    for rec in summary:
        print(
            f"{rec['scheme']:<19} | {rec['snr_db']:>6.1f} | {rec['n_mc']:>4d} | "
            f"{rec['mean_nmse_ha']:.3e} | {rec['std_nmse_ha']:.3e} | "
            f"{rec['mean_nmse_h']:.3e} | {rec['std_nmse_h']:.3e}"
        )
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
