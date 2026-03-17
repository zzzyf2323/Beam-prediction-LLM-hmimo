"""Monte Carlo static-evaluation smoke test."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_static_mc_eval_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "hmimo_eval_static.py"
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(
        """
seed: 9
n_mc: 2
snr_db_list: [0.0, 5.0]
channel:
  seed: 9
  rx_dims: [2, 2]
  tx_dims: [2, 2]
  d_over_lambda: 0.5
  n_clusters: 2
  cluster_spread: 0.8
probing:
  n_slots: 3
  n_rx_rf: 1
  n_tx_rf: 1
  no_repeat: true
grouping:
  rx_block_shape: [1, 1]
  tx_block_shape: [1, 1]
group_lasso:
  lam: 0.01
  max_iter: 40
  tol: 1.0e-6
group_sbl:
  max_iter: 15
  tol: 1.0e-5
  damping: 0.4
""".strip()
    )

    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg), "--output-dir", str(out_dir)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    json_path = out_dir / "hmimo_static_results.json"
    csv_path = out_dir / "hmimo_static_summary.csv"
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text())
    trials = payload["trial_results"]
    summary = payload["summary"]

    expected_schemes = {
        "fpws+group_lasso",
        "fpws+group_sbl",
        "dft+group_lasso",
        "dft+group_sbl",
    }
    expected_snrs = {0.0, 5.0}

    for snr in expected_snrs:
        schemes_at_snr = {r["scheme"] for r in trials if float(r["snr_db"]) == snr}
        assert expected_schemes.issubset(schemes_at_snr)

    for rec in summary:
        assert np.isfinite(float(rec["mean_nmse_ha"]))
        assert np.isfinite(float(rec["mean_nmse_h"]))
        assert np.isfinite(float(rec["std_nmse_ha"]))
        assert np.isfinite(float(rec["std_nmse_h"]))

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(summary)
    pair_set = {(row["scheme"], float(row["snr_db"])) for row in rows}
    for snr in expected_snrs:
        for scheme in expected_schemes:
            assert (scheme, snr) in pair_set
