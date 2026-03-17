# HMIMO experiment map

## Script/config/output mapping

### 1) Static main benchmark
- Script: `scripts/hmimo_eval_static.py`
- Config: `configs/hmimo_static_small.yaml`
- Outputs:
  - `hmimo_static_results.json`
  - `hmimo_static_summary.csv`

### 2) Dynamic baseline benchmark (non-LLM temporal)
- Script: `scripts/hmimo_eval_dynamic.py`
- Config: `configs/hmimo_dynamic_small.yaml`
- Outputs:
  - `hmimo_dynamic_results.json`
  - `hmimo_dynamic_summary.csv`

### 3) Dynamic LLM benchmark
- Script: `scripts/hmimo_eval_dynamic_llm.py`
- Config: `configs/hmimo_dynamic_llm_small.yaml`
- Outputs:
  - `hmimo_dynamic_llm_results.json`
  - `hmimo_dynamic_llm_summary.csv`

### 4) Dynamic LLM ablation benchmark
- Script: `scripts/hmimo_ablate_dynamic_llm.py`
- Config: `configs/hmimo_dynamic_llm_ablation_small.yaml`
- Outputs:
  - `hmimo_dynamic_llm_ablation_results.json`
  - `hmimo_dynamic_llm_ablation_summary.csv`

### 5) Paper figure export
- Script: `scripts/hmimo_export_paper_figures.py`
- Inputs: JSON outputs from items (1)-(4)
- Outputs (folder): `outputs/paper_figures`

### 6) Paper table export
- Script: `scripts/hmimo_export_paper_tables.py`
- Inputs: JSON outputs from items (1)-(4)
- Outputs (folder): `outputs/paper_tables`
