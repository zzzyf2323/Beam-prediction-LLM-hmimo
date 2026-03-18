"""Smoke test for static-eval script harness."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_static_eval_script_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "hmimo_eval_static.py"
    config = repo_root / "configs" / "hmimo_static_small.yaml"
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(config), "--output-dir", str(out_dir)],
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
    results = payload["trial_results"]
    summary = payload["summary"]
    assert len(results) > 0
    assert len(summary) > 0

    schemes = {r["scheme"] for r in results}
    expected = {
        "fpws+group_lasso",
        "fpws+group_sbl",
        "dft+group_lasso",
        "dft+group_sbl",
    }
    assert expected.issubset(schemes)

    for r in results:
        assert np.isfinite(float(r["nmse_ha"]))
        assert np.isfinite(float(r["nmse_h"]))

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(summary)
    csv_schemes = {row["scheme"] for row in rows}
    assert expected.issubset(csv_schemes)
