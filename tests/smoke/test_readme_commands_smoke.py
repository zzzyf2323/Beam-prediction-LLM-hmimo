"""Lightweight sync check for README quickstart script/config references."""

from pathlib import Path


def test_readme_quickstart_references_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text()

    required_paths = [
        "scripts/hmimo_eval_static.py",
        "scripts/hmimo_eval_dynamic.py",
        "scripts/hmimo_eval_dynamic_llm.py",
        "scripts/hmimo_ablate_dynamic_llm.py",
        "scripts/hmimo_export_paper_figures.py",
        "scripts/hmimo_export_paper_tables.py",
        "configs/hmimo_static_small.yaml",
        "configs/hmimo_dynamic_small.yaml",
        "configs/hmimo_dynamic_llm_small.yaml",
        "configs/hmimo_dynamic_llm_ablation_small.yaml",
    ]

    for rel in required_paths:
        assert rel in readme
        assert (root / rel).exists()
