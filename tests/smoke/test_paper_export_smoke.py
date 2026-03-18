"""Smoke test for HMIMO paper export scripts."""

import subprocess
import sys
from pathlib import Path


def _run(cmd, cwd: Path):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_paper_export_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    out_static = tmp_path / "static"
    out_dynamic = tmp_path / "dynamic"
    out_dyn_llm = tmp_path / "dyn_llm"
    out_abl = tmp_path / "ablation"
    out_export = tmp_path / "exports"

    _run(
        [
            sys.executable,
            "scripts/hmimo_eval_static.py",
            "--config",
            "configs/hmimo_static_small.yaml",
            "--output-dir",
            str(out_static),
        ],
        repo_root,
    )
    _run(
        [
            sys.executable,
            "scripts/hmimo_eval_dynamic.py",
            "--config",
            "configs/hmimo_dynamic_small.yaml",
            "--output-dir",
            str(out_dynamic),
        ],
        repo_root,
    )
    _run(
        [
            sys.executable,
            "scripts/hmimo_eval_dynamic_llm.py",
            "--config",
            "configs/hmimo_dynamic_llm_small.yaml",
            "--output-dir",
            str(out_dyn_llm),
        ],
        repo_root,
    )
    _run(
        [
            sys.executable,
            "scripts/hmimo_ablate_dynamic_llm.py",
            "--config",
            "configs/hmimo_dynamic_llm_ablation_small.yaml",
            "--output-dir",
            str(out_abl),
        ],
        repo_root,
    )

    _run(
        [
            sys.executable,
            "scripts/hmimo_export_paper_figures.py",
            "--static-json",
            str(out_static / "hmimo_static_results.json"),
            "--dynamic-json",
            str(out_dynamic / "hmimo_dynamic_results.json"),
            "--dynamic-llm-json",
            str(out_dyn_llm / "hmimo_dynamic_llm_results.json"),
            "--ablation-json",
            str(out_abl / "hmimo_dynamic_llm_ablation_results.json"),
            "--output-root",
            str(out_export),
        ],
        repo_root,
    )

    _run(
        [
            sys.executable,
            "scripts/hmimo_export_paper_tables.py",
            "--static-json",
            str(out_static / "hmimo_static_results.json"),
            "--dynamic-json",
            str(out_dynamic / "hmimo_dynamic_results.json"),
            "--dynamic-llm-json",
            str(out_dyn_llm / "hmimo_dynamic_llm_results.json"),
            "--ablation-json",
            str(out_abl / "hmimo_dynamic_llm_ablation_results.json"),
            "--output-root",
            str(out_export),
        ],
        repo_root,
    )

    fig_dir = out_export / "paper_figures"
    tab_dir = out_export / "paper_tables"

    expected_fig = [
        "fig_static_nmse_vs_snr.png",
        "fig_static_nmse_vs_snr.pdf",
        "fig_dynamic_temporal_compare.png",
        "fig_dynamic_temporal_compare.pdf",
        "fig_dynamic_llm_ablation.png",
        "fig_dynamic_llm_ablation.pdf",
    ]
    for name in expected_fig:
        assert (fig_dir / name).exists()

    expected_tbl = [
        "table_static_summary.csv",
        "table_dynamic_summary.csv",
        "table_dynamic_llm_summary.csv",
        "table_dynamic_llm_ablation_summary.csv",
    ]
    for name in expected_tbl:
        assert (tab_dir / name).exists()
