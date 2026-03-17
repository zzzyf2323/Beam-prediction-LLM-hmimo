<div align="center">

  <h2><b>HMIMO Research Codebase (FPWS + Operator-Form Recovery)</b></h2>
</div>

## HMIMO overview (current repository status)
This repository now includes an HMIMO-focused pipeline with reproducible static and dynamic experiments:

1. **Static HMIMO benchmark** (FPWS vs DFT probing; Group-LASSO and Group-SBL).  
2. **Dynamic HMIMO baseline benchmark** (non-LLM temporal warm-start).  
3. **Dynamic HMIMO LLM-prior benchmark** (LLM-assisted temporal prior).  
4. **Dynamic HMIMO LLM ablation benchmark** (controlled prior-input ablations).  
5. **Paper export scripts** for figures and compact tables.

> Important: the LLM path is **prior-only**. The final channel estimator remains **Group-SBL** (or Group-LASSO in non-LLM runs). The physical HMIMO operator backbone is preserved.

## Main runnable scripts
- `scripts/hmimo_eval_static.py`: static Monte Carlo benchmark.
- `scripts/hmimo_eval_dynamic.py`: dynamic non-LLM temporal baseline benchmark.
- `scripts/hmimo_eval_dynamic_llm.py`: dynamic benchmark with no-temporal / non-LLM / LLM-prior Group-SBL.
- `scripts/hmimo_ablate_dynamic_llm.py`: controlled dynamic LLM-prior ablations.
- `scripts/hmimo_export_paper_figures.py`: exports paper-style PNG/PDF figures from JSON outputs.
- `scripts/hmimo_export_paper_tables.py`: exports compact paper-style CSV tables from JSON outputs.

## Minimal quickstart
Install dependencies:
```bash
pip install -r requirements.txt
```

Run static benchmark:
```bash
python scripts/hmimo_eval_static.py \
  --config configs/hmimo_static_small.yaml \
  --output-dir outputs/static_small
```

Run dynamic non-LLM benchmark:
```bash
python scripts/hmimo_eval_dynamic.py \
  --config configs/hmimo_dynamic_small.yaml \
  --output-dir outputs/dynamic_small
```

Run dynamic LLM-prior benchmark:
```bash
python scripts/hmimo_eval_dynamic_llm.py \
  --config configs/hmimo_dynamic_llm_small.yaml \
  --output-dir outputs/dynamic_llm_small
```

Run dynamic LLM ablation benchmark:
```bash
python scripts/hmimo_ablate_dynamic_llm.py \
  --config configs/hmimo_dynamic_llm_ablation_small.yaml \
  --output-dir outputs/dynamic_llm_ablation_small
```

Export paper figures and tables:
```bash
python scripts/hmimo_export_paper_figures.py \
  --static-json outputs/static_small/hmimo_static_results.json \
  --dynamic-json outputs/dynamic_small/hmimo_dynamic_results.json \
  --dynamic-llm-json outputs/dynamic_llm_small/hmimo_dynamic_llm_results.json \
  --ablation-json outputs/dynamic_llm_ablation_small/hmimo_dynamic_llm_ablation_results.json \
  --output-root outputs

python scripts/hmimo_export_paper_tables.py \
  --static-json outputs/static_small/hmimo_static_results.json \
  --dynamic-json outputs/dynamic_small/hmimo_dynamic_results.json \
  --dynamic-llm-json outputs/dynamic_llm_small/hmimo_dynamic_llm_results.json \
  --ablation-json outputs/dynamic_llm_ablation_small/hmimo_dynamic_llm_ablation_results.json \
  --output-root outputs
```

## Reproducibility and experiment docs
- `docs/reproducibility.md`
- `docs/experiment_map.md`
- `docs/results_inventory.md`
- `configs/README.md`

## Legacy note
Original beam-prediction assets and scripts are retained in this repository for archival/reference continuity.
