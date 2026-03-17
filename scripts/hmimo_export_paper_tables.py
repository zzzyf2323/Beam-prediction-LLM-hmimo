"""Export compact paper tables from existing HMIMO benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _write_table(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-json", required=True)
    parser.add_argument("--dynamic-json", required=True)
    parser.add_argument("--dynamic-llm-json", required=True)
    parser.add_argument("--ablation-json", required=True)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_root) / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    static_payload = _load_json(Path(args.static_json))
    dynamic_payload = _load_json(Path(args.dynamic_json))
    llm_payload = _load_json(Path(args.dynamic_llm_json))
    ablation_payload = _load_json(Path(args.ablation_json))

    fields = ["scheme", "snr_db", "mean_nmse_ha", "std_nmse_ha", "mean_nmse_h", "std_nmse_h"]

    _write_table(out_dir / "table_static_summary.csv", list(static_payload["summary"]), fields)
    _write_table(out_dir / "table_dynamic_summary.csv", list(dynamic_payload["summary"]), fields)
    _write_table(out_dir / "table_dynamic_llm_summary.csv", list(llm_payload["summary"]), fields)
    _write_table(out_dir / "table_dynamic_llm_ablation_summary.csv", list(ablation_payload["summary"]), fields)

    print(f"Saved tables to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
