"""Dynamic HMIMO evaluation with non-LLM temporal warm-start baselines."""

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
from hmimo.priors.temporal_prior import temporal_warm_start_from_prev
from hmimo.probing.fpws_selection import build_fpws_selection_contexts
from hmimo.simulation.channel import build_propagating_fpws_bases
from hmimo.simulation.dynamic_channel import generate_dynamic_hmimo_sequence

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


def _aggregate(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, float], List[Dict[str, object]]] = {}
    for r in records:
        key = (str(r["scheme"]), float(r["snr_db"]))
        buckets.setdefault(key, []).append(r)

    summary = []
    for (scheme, snr_db), rows in sorted(buckets.items(), key=lambda x: (x[0][1], x[0][0])):
        a = np.asarray([float(x["nmse_ha"]) for x in rows], dtype=float)
        b = np.asarray([float(x["nmse_h"]) for x in rows], dtype=float)
        summary.append(
            {
                "scheme": scheme,
                "snr_db": snr_db,
                "count": len(rows),
                "mean_nmse_ha": float(np.mean(a)),
                "std_nmse_ha": float(np.std(a)),
                "mean_nmse_h": float(np.mean(b)),
                "std_nmse_h": float(np.std(b)),
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
    seed = int(cfg["seed"])

    all_records: List[Dict[str, object]] = []

    for trial in range(n_mc):
        dcfg = dict(cfg["dynamic_channel"])
        dcfg["seed"] = int(dcfg.get("seed", seed)) + trial
        seq = generate_dynamic_hmimo_sequence(dcfg)

        ha_seq = seq["ha_seq"]
        h_seq = seq["h_seq"]
        rx_pairs = seq["rx_mode_pairs"]
        tx_pairs = seq["tx_mode_pairs"]
        shape = ha_seq[0].shape

        psi_r_prop, psi_s_prop = build_propagating_fpws_bases(
            rx_dims=tuple(seq["rx_dims"]),
            tx_dims=tuple(seq["tx_dims"]),
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
        contexts = build_fpws_selection_contexts(
            n_rx_modes=rx_pairs.shape[0],
            n_tx_modes=tx_pairs.shape[0],
            n_slots=int(probing["n_slots"]),
            n_rx_rf=int(probing["n_rx_rf"]),
            n_tx_rf=int(probing["n_tx_rf"]),
            seed=seed + 1000 * trial,
            no_repeat=bool(probing.get("no_repeat", True)),
        )

        fwd = lambda h: forward_operator(h, contexts)
        adj = lambda r: adjoint_operator(r, contexts, ha_shape=shape)

        prev_state: Dict[str, Dict[str, object]] = {
            "sbl_no": {"ha": None, "gamma": None},
            "sbl_temporal": {"ha": None, "gamma": None},
            "lasso_no": {"ha": None},
            "lasso_temporal": {"ha": None},
        }

        for snr_db in snr_list:
            for t, (ha_true, h_true) in enumerate(zip(ha_seq, h_seq)):
                power = float(np.mean(np.abs(ha_true) ** 2))
                noise_var = power / (10.0 ** (snr_db / 10.0))
                nrng = np.random.default_rng(seed + 10000 * trial + int(100 * snr_db) + t)
                noise = np.sqrt(noise_var / 2.0) * (
                    nrng.normal(size=fwd(ha_true).shape) + 1j * nrng.normal(size=fwd(ha_true).shape)
                )
                y = fwd(ha_true) + noise

                # Group-SBL without temporal warm-start.
                out_sbl_no = solve_group_sbl(
                    forward_op=fwd,
                    adjoint_op=adj,
                    y=y,
                    shape=shape,
                    groups=groups,
                    noise_var=max(noise_var, 1e-10),
                    max_iter=int(cfg["group_sbl"]["max_iter"]),
                    tol=float(cfg["group_sbl"]["tol"]),
                    damping=float(cfg["group_sbl"]["damping"]),
                    gamma_init=1.0,
                )
                ha_hat = out_sbl_no["ha_hat"]
                h_hat = psi_r_prop @ ha_hat @ psi_s_prop.conj().T
                all_records.append(
                    {
                        "trial": trial,
                        "frame": t,
                        "snr_db": snr_db,
                        "scheme": "fpws+group_sbl_no_temporal",
                        "nmse_ha": nmse_ha(ha_hat, ha_true),
                        "nmse_h": nmse_h(h_hat, h_true),
                    }
                )
                prev_state["sbl_no"]["ha"] = ha_hat
                prev_state["sbl_no"]["gamma"] = out_sbl_no.get("gamma_group")

                # Group-SBL with non-LLM temporal warm-start.
                warm_sbl = temporal_warm_start_from_prev(
                    prev_ha_hat=prev_state["sbl_temporal"].get("ha"),
                    groups=groups,
                    prev_gamma=prev_state["sbl_temporal"].get("gamma"),
                    alpha=float(cfg["temporal_prior"]["alpha"]),
                )
                gamma_init_scalar = float(np.mean(np.asarray(warm_sbl["gamma_group_init"], dtype=float)))
                out_sbl_temporal = solve_group_sbl(
                    forward_op=fwd,
                    adjoint_op=adj,
                    y=y,
                    shape=shape,
                    groups=groups,
                    noise_var=max(noise_var, 1e-10),
                    max_iter=int(cfg["group_sbl"]["max_iter"]),
                    tol=float(cfg["group_sbl"]["tol"]),
                    damping=float(cfg["group_sbl"]["damping"]),
                    gamma_init=max(gamma_init_scalar, 1e-8),
                )
                ha_hat = out_sbl_temporal["ha_hat"]
                h_hat = psi_r_prop @ ha_hat @ psi_s_prop.conj().T
                all_records.append(
                    {
                        "trial": trial,
                        "frame": t,
                        "snr_db": snr_db,
                        "scheme": "fpws+group_sbl_temporal",
                        "nmse_ha": nmse_ha(ha_hat, ha_true),
                        "nmse_h": nmse_h(h_hat, h_true),
                    }
                )
                prev_state["sbl_temporal"]["ha"] = ha_hat
                prev_state["sbl_temporal"]["gamma"] = out_sbl_temporal.get("gamma_group")

                # Group-LASSO without temporal warm-start.
                out_lasso_no = solve_group_lasso(
                    forward_op=fwd,
                    adjoint_op=adj,
                    y=y,
                    shape=shape,
                    groups=groups,
                    lam=float(cfg["group_lasso"]["lam"]),
                    max_iter=int(cfg["group_lasso"]["max_iter"]),
                    tol=float(cfg["group_lasso"]["tol"]),
                )
                ha_hat = out_lasso_no["ha_hat"]
                h_hat = psi_r_prop @ ha_hat @ psi_s_prop.conj().T
                all_records.append(
                    {
                        "trial": trial,
                        "frame": t,
                        "snr_db": snr_db,
                        "scheme": "fpws+group_lasso_no_temporal",
                        "nmse_ha": nmse_ha(ha_hat, ha_true),
                        "nmse_h": nmse_h(h_hat, h_true),
                    }
                )
                prev_state["lasso_no"]["ha"] = ha_hat

                # Group-LASSO with non-LLM temporal warm-start via lam scaling.
                warm_lasso = temporal_warm_start_from_prev(
                    prev_ha_hat=prev_state["lasso_temporal"].get("ha"),
                    groups=groups,
                    alpha=float(cfg["temporal_prior"]["alpha"]),
                )
                lam_scaled = float(cfg["group_lasso"]["lam"]) * float(warm_lasso["lasso_lam_scale"])
                out_lasso_temporal = solve_group_lasso(
                    forward_op=fwd,
                    adjoint_op=adj,
                    y=y,
                    shape=shape,
                    groups=groups,
                    lam=lam_scaled,
                    max_iter=int(cfg["group_lasso"]["max_iter"]),
                    tol=float(cfg["group_lasso"]["tol"]),
                )
                ha_hat = out_lasso_temporal["ha_hat"]
                h_hat = psi_r_prop @ ha_hat @ psi_s_prop.conj().T
                all_records.append(
                    {
                        "trial": trial,
                        "frame": t,
                        "snr_db": snr_db,
                        "scheme": "fpws+group_lasso_temporal",
                        "nmse_ha": nmse_ha(ha_hat, ha_true),
                        "nmse_h": nmse_h(h_hat, h_true),
                    }
                )
                prev_state["lasso_temporal"]["ha"] = ha_hat

    summary = _aggregate(all_records)

    json_path = out_dir / "hmimo_dynamic_results.json"
    csv_path = out_dir / "hmimo_dynamic_summary.csv"

    json_path.write_text(json.dumps({"config": cfg, "trial_results": all_records, "summary": summary}, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scheme", "snr_db", "count", "mean_nmse_ha", "std_nmse_ha", "mean_nmse_h", "std_nmse_h"],
        )
        writer.writeheader()
        writer.writerows(summary)

    print("Dynamic HMIMO summary")
    print("scheme                         | snr_db | count | mean_nmse_ha | std_nmse_ha | mean_nmse_h | std_nmse_h")
    for rec in summary:
        print(
            f"{rec['scheme']:<30} | {rec['snr_db']:>6.1f} | {rec['count']:>5d} | "
            f"{rec['mean_nmse_ha']:.3e} | {rec['std_nmse_ha']:.3e} | "
            f"{rec['mean_nmse_h']:.3e} | {rec['std_nmse_h']:.3e}"
        )
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
