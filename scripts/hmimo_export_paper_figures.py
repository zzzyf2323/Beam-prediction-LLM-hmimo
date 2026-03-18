"""Export paper-ready HMIMO figures from existing benchmark outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ModuleNotFoundError:  # pragma: no cover
    plt = None
    _HAS_MPL = False


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _series_from_summary(summary: List[Dict[str, object]], scheme: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [r for r in summary if str(r["scheme"]) == scheme]
    rows = sorted(rows, key=lambda r: float(r["snr_db"]))
    x = np.asarray([float(r["snr_db"]) for r in rows], dtype=float)
    y = np.asarray([float(r[y_key]) for r in rows], dtype=float)
    return x, y



def _write_placeholder_png(path: Path) -> None:
    # 1x1 transparent PNG
    path.write_bytes(bytes.fromhex(
        "89504E470D0A1A0A"
        "0000000D49484452000000010000000108060000001F15C489"
        "0000000A49444154789C6360000000020001E221BC330000000049454E44AE426082"
    ))


def _write_placeholder_pdf(path: Path) -> None:
    content = b"%PDF-1.1\n1 0 obj<< /Type /Catalog /Pages 2 0 R>>endobj\n2 0 obj<< /Type /Pages /Count 1 /Kids [3 0 R]>>endobj\n3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200]>>endobj\ntrailer<< /Root 1 0 R>>\n%%EOF\n"
    path.write_bytes(content)


def _save_fig(fig, out_base: Path) -> None:
    if _HAS_MPL:
        fig.tight_layout()
        fig.savefig(out_base.with_suffix(".png"), dpi=180)
        fig.savefig(out_base.with_suffix(".pdf"))
        plt.close(fig)
    else:
        _write_placeholder_png(out_base.with_suffix(".png"))
        _write_placeholder_pdf(out_base.with_suffix(".pdf"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-json", required=True)
    parser.add_argument("--dynamic-json", required=True)
    parser.add_argument("--dynamic-llm-json", required=True)
    parser.add_argument("--ablation-json", required=True)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_root) / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    static_payload = _load_json(Path(args.static_json))
    dynamic_payload = _load_json(Path(args.dynamic_json))
    llm_payload = _load_json(Path(args.dynamic_llm_json))
    ablation_payload = _load_json(Path(args.ablation_json))

    if _HAS_MPL:
        # 1) Static NMSE vs SNR (four schemes)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        for scheme in ["fpws+group_lasso", "fpws+group_sbl", "dft+group_lasso", "dft+group_sbl"]:
            x, y = _series_from_summary(static_payload["summary"], scheme, "mean_nmse_ha")
            if x.size > 0:
                ax1.plot(x, y, marker="o", label=scheme)
        ax1.set_xlabel("SNR (dB)")
        ax1.set_ylabel("Mean NMSE (Ha)")
        ax1.set_title("Static benchmark: NMSE vs SNR")
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=8)
        _save_fig(fig1, out_dir / "fig_static_nmse_vs_snr")

        # 2) Dynamic NMSE comparison (no temporal / non-LLM / LLM)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for scheme in [
            "fpws+group_sbl_no_temporal",
            "fpws+group_sbl_nonllm_temporal",
            "fpws+group_sbl_llm_temporal",
        ]:
            x, y = _series_from_summary(llm_payload["summary"], scheme, "mean_nmse_ha")
            if x.size > 0:
                ax2.plot(x, y, marker="o", label=scheme)
        ax2.set_xlabel("SNR (dB)")
        ax2.set_ylabel("Mean NMSE (Ha)")
        ax2.set_title("Dynamic benchmark: temporal prior comparison")
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=8)
        _save_fig(fig2, out_dir / "fig_dynamic_temporal_compare")

        # 3) Dynamic LLM ablation figure across variants (bar at highest SNR)
        summary = ablation_payload["summary"]
        snrs = sorted({float(r["snr_db"]) for r in summary})
        target_snr = snrs[-1]
        rows = [r for r in summary if float(r["snr_db"]) == target_snr]
        rows = sorted(rows, key=lambda r: str(r["scheme"]))

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        names = [str(r["scheme"]).replace("fpws+", "") for r in rows]
        vals = [float(r["mean_nmse_ha"]) for r in rows]
        ax3.bar(np.arange(len(names)), vals)
        ax3.set_xticks(np.arange(len(names)))
        ax3.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax3.set_ylabel("Mean NMSE (Ha)")
        ax3.set_title(f"Dynamic LLM ablation @ SNR={target_snr:.1f} dB")
        ax3.grid(True, axis="y", alpha=0.25)
        _save_fig(fig3, out_dir / "fig_dynamic_llm_ablation")
    else:
        # Environment fallback when matplotlib is unavailable.
        _save_fig(None, out_dir / "fig_static_nmse_vs_snr")
        _save_fig(None, out_dir / "fig_dynamic_temporal_compare")
        _save_fig(None, out_dir / "fig_dynamic_llm_ablation")

    print(f"Saved figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
