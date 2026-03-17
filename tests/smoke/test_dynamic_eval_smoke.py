"""Smoke test for dynamic HMIMO evaluation script."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_dynamic_eval_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "hmimo_eval_dynamic.py"
    cfg = repo_root / "configs" / "hmimo_dynamic_small.yaml"
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg), "--output-dir", str(out_dir)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    json_path = out_dir / "hmimo_dynamic_results.json"
    csv_path = out_dir / "hmimo_dynamic_summary.csv"
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text())
    trials = payload["trial_results"]
    summary = payload["summary"]

    expected = {
        "fpws+group_sbl_no_temporal",
        "fpws+group_sbl_temporal",
        "fpws+group_lasso_no_temporal",
        "fpws+group_lasso_temporal",
    }
    schemes = {r["scheme"] for r in trials}
    assert expected.issubset(schemes)

    for rec in trials:
        assert np.isfinite(float(rec["nmse_ha"]))
        assert np.isfinite(float(rec["nmse_h"]))

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(summary)
    csv_schemes = {r["scheme"] for r in rows}
    assert expected.issubset(csv_schemes)
    for r in rows:
        assert np.isfinite(float(r["mean_nmse_ha"]))
        assert np.isfinite(float(r["mean_nmse_h"]))
