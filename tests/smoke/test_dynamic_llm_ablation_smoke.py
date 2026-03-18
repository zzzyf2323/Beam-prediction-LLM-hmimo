"""Smoke test for dynamic LLM-prior ablation harness."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_dynamic_llm_ablation_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "hmimo_ablate_dynamic_llm.py"
    cfg = repo_root / "configs" / "hmimo_dynamic_llm_ablation_small.yaml"
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg), "--output-dir", str(out_dir)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    json_path = out_dir / "hmimo_dynamic_llm_ablation_results.json"
    csv_path = out_dir / "hmimo_dynamic_llm_ablation_summary.csv"
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text())
    trials = payload["trial_results"]
    summary = payload["summary"]

    expected = {
        "fpws+group_sbl_no_temporal",
        "fpws+group_sbl_nonllm_temporal",
        "fpws+group_sbl_llm_full",
        "fpws+group_sbl_llm_no_metadata",
        "fpws+group_sbl_llm_no_prev_gamma",
        "fpws+group_sbl_llm_no_prev_energy",
        "fpws+group_sbl_llm_short_history",
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
    for row in rows:
        assert np.isfinite(float(row["mean_nmse_ha"]))
        assert np.isfinite(float(row["mean_nmse_h"]))
