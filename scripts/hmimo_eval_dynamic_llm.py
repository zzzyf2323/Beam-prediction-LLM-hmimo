"""Dynamic HMIMO evaluation with optional LLM-assisted temporal prior."""

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

from hmimo.estimators.group_sbl import solve_group_sbl
from hmimo.evaluation.metrics import nmse_h, nmse_ha
from hmimo.grouping.groups import build_rx_tx_groups
from hmimo.physics.operators import adjoint_operator, forward_operator
from hmimo.priors.temporal_prior import temporal_warm_start_from_prev
from hmimo.probing.fpws_selection import build_fpws_selection_contexts
from hmimo.simulation.channel import build_propagating_fpws_bases
from hmimo.simulation.dynamic_channel import generate_dynamic_hmimo_sequence
from models.temporal_prior.hmimo_llm_prior import HMIMOLLMTemporalPrior, LLMTemporalPriorConfig

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
        buckets.setdefault((str(r["scheme"]), float(r["snr_db"])), []).append(r)

    out = []
    for (scheme, snr), rows in sorted(buckets.items(), key=lambda x: (x[0][1], x[0][0])):
        a = np.asarray([float(r["nmse_ha"]) for r in rows], dtype=float)
        b = np.asarray([float(r["nmse_h"]) for r in rows], dtype=float)
        out.append(
            {
                "scheme": scheme,
                "snr_db": snr,
                "count": len(rows),
                "mean_nmse_ha": float(np.mean(a)),
                "std_nmse_ha": float(np.std(a)),
                "mean_nmse_h": float(np.mean(b)),
                "std_nmse_h": float(np.std(b)),
            }
        )
    return out


def _run_sbl(
    fwd,
    adj,
    y,
    shape,
    groups,
    noise_var: float,
    cfg: Dict[str, object],
    gamma_init_scalar: float,
):
    return solve_group_sbl(
        forward_op=fwd,
        adjoint_op=adj,
        y=y,
        shape=shape,
        groups=groups,
        noise_var=max(noise_var, 1e-10),
        max_iter=int(cfg["max_iter"]),
        tol=float(cfg["tol"]),
        damping=float(cfg["damping"]),
        gamma_init=max(gamma_init_scalar, 1e-8),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = LLMTemporalPriorConfig(
        enabled=bool(cfg["llm_prior"].get("enabled", True)),
        history_len=int(cfg["llm_prior"].get("history_len", 3)),
        fusion_dim=int(cfg["llm_prior"].get("fusion_dim", 32)),
        seed=int(cfg["llm_prior"].get("seed", cfg["seed"])),
    )
    llm_prior = HMIMOLLMTemporalPrior(llm_cfg)

    snr_list = [float(x) for x in cfg["snr_db_list"]]
    n_mc = int(cfg.get("n_mc", 1))
    base_seed = int(cfg["seed"])

    records: List[Dict[str, object]] = []

    for trial in range(n_mc):
        dcfg = dict(cfg["dynamic_channel"])
        dcfg["seed"] = int(dcfg.get("seed", base_seed)) + trial
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

        contexts = build_fpws_selection_contexts(
            n_rx_modes=rx_pairs.shape[0],
            n_tx_modes=tx_pairs.shape[0],
            n_slots=int(cfg["probing"]["n_slots"]),
            n_rx_rf=int(cfg["probing"]["n_rx_rf"]),
            n_tx_rf=int(cfg["probing"]["n_tx_rf"]),
            seed=base_seed + 1000 * trial,
            no_repeat=bool(cfg["probing"].get("no_repeat", True)),
        )
        fwd = lambda h: forward_operator(h, contexts)
        adj = lambda r: adjoint_operator(r, contexts, ha_shape=shape)

        prev_no = {"ha": None, "gamma": None}
        prev_warm = {"ha": None, "gamma": None}
        prev_llm = {"ha": None, "gamma": None}
        llm_history: List[Dict[str, np.ndarray | float]] = []

        for snr_db in snr_list:
            for frame_idx, (ha_true, h_true) in enumerate(zip(ha_seq, h_seq)):
                power = float(np.mean(np.abs(ha_true) ** 2))
                noise_var = power / (10.0 ** (snr_db / 10.0))
                nrng = np.random.default_rng(base_seed + 10000 * trial + int(100 * snr_db) + frame_idx)
                noise = np.sqrt(noise_var / 2.0) * (
                    nrng.normal(size=fwd(ha_true).shape) + 1j * nrng.normal(size=fwd(ha_true).shape)
                )
                y = fwd(ha_true) + noise

                out_no = _run_sbl(
                    fwd, adj, y, shape, groups, noise_var, cfg["group_sbl"], gamma_init_scalar=1.0
                )
                ha_no = out_no["ha_hat"]
                h_no = psi_r_prop @ ha_no @ psi_s_prop.conj().T
                records.append(
                    {
                        "trial": trial,
                        "frame": frame_idx,
                        "snr_db": snr_db,
                        "scheme": "fpws+group_sbl_no_temporal",
                        "nmse_ha": nmse_ha(ha_no, ha_true),
                        "nmse_h": nmse_h(h_no, h_true),
                    }
                )
                prev_no = {"ha": ha_no, "gamma": out_no.get("gamma_group")}

                warm = temporal_warm_start_from_prev(
                    prev_ha_hat=prev_warm.get("ha"),
                    groups=groups,
                    prev_gamma=prev_warm.get("gamma"),
                    alpha=float(cfg["temporal_prior"]["alpha"]),
                )
                out_warm = _run_sbl(
                    fwd,
                    adj,
                    y,
                    shape,
                    groups,
                    noise_var,
                    cfg["group_sbl"],
                    gamma_init_scalar=float(np.mean(np.asarray(warm["gamma_group_init"], dtype=float))),
                )
                ha_warm = out_warm["ha_hat"]
                h_warm = psi_r_prop @ ha_warm @ psi_s_prop.conj().T
                records.append(
                    {
                        "trial": trial,
                        "frame": frame_idx,
                        "snr_db": snr_db,
                        "scheme": "fpws+group_sbl_nonllm_temporal",
                        "nmse_ha": nmse_ha(ha_warm, ha_true),
                        "nmse_h": nmse_h(h_warm, h_true),
                    }
                )
                prev_warm = {"ha": ha_warm, "gamma": out_warm.get("gamma_group")}

                fallback = np.asarray(warm["gamma_group_init"], dtype=float)
                meta = {
                    "n_groups": float(len(groups)),
                    "rf_budget": float(int(cfg["probing"]["n_rx_rf"]) + int(cfg["probing"]["n_tx_rf"])),
                }
                llm_hint = llm_prior.infer_prior(
                    history=llm_history,
                    metadata=meta,
                    fallback_gamma=fallback,
                )
                out_llm = _run_sbl(
                    fwd,
                    adj,
                    y,
                    shape,
                    groups,
                    noise_var,
                    cfg["group_sbl"],
                    gamma_init_scalar=float(np.mean(np.asarray(llm_hint["gamma_init"], dtype=float))),
                )
                ha_llm = out_llm["ha_hat"]
                h_llm = psi_r_prop @ ha_llm @ psi_s_prop.conj().T
                records.append(
                    {
                        "trial": trial,
                        "frame": frame_idx,
                        "snr_db": snr_db,
                        "scheme": "fpws+group_sbl_llm_temporal",
                        "nmse_ha": nmse_ha(ha_llm, ha_true),
                        "nmse_h": nmse_h(h_llm, h_true),
                    }
                )
                prev_llm = {"ha": ha_llm, "gamma": out_llm.get("gamma_group")}

                gamma_arr = np.asarray(out_llm.get("gamma_group"), dtype=float)
                energy = np.asarray([float(np.mean(np.abs(np.asarray(ha_llm).reshape(-1)[np.asarray(g)] ** 2))) for g in groups])
                active_prob = np.clip(gamma_arr / (np.max(gamma_arr) + 1e-12), 0.0, 1.0)
                llm_history.append(
                    {
                        "gamma": gamma_arr,
                        "energy": energy,
                        "active_prob": active_prob,
                        "snr_db": float(snr_db),
                    }
                )

    summary = _aggregate(records)

    json_path = out_dir / "hmimo_dynamic_llm_results.json"
    csv_path = out_dir / "hmimo_dynamic_llm_summary.csv"

    json_path.write_text(json.dumps({"config": cfg, "trial_results": records, "summary": summary}, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scheme", "snr_db", "count", "mean_nmse_ha", "std_nmse_ha", "mean_nmse_h", "std_nmse_h"],
        )
        writer.writeheader()
        writer.writerows(summary)

    print("Dynamic HMIMO LLM-prior summary")
    for rec in summary:
        print(
            f"{rec['scheme']:<32} | SNR={rec['snr_db']:>5.1f} | "
            f"NMSE(Ha)={rec['mean_nmse_ha']:.3e}±{rec['std_nmse_ha']:.3e} | "
            f"NMSE(H)={rec['mean_nmse_h']:.3e}±{rec['std_nmse_h']:.3e}"
        )
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
