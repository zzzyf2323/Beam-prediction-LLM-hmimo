# HMIMO reproducibility guide

## Environment assumptions
- Python environment with dependencies from `requirements.txt`.
- Scripts are executed from repository root.
- All HMIMO runs are configuration + seed driven.

## Recommended execution order
1. Static benchmark
   - `python scripts/hmimo_eval_static.py --config configs/hmimo_static_small.yaml --output-dir outputs/static_small`
2. Dynamic non-LLM benchmark
   - `python scripts/hmimo_eval_dynamic.py --config configs/hmimo_dynamic_small.yaml --output-dir outputs/dynamic_small`
3. Dynamic LLM benchmark
   - `python scripts/hmimo_eval_dynamic_llm.py --config configs/hmimo_dynamic_llm_small.yaml --output-dir outputs/dynamic_llm_small`
4. Dynamic LLM ablation benchmark
   - `python scripts/hmimo_ablate_dynamic_llm.py --config configs/hmimo_dynamic_llm_ablation_small.yaml --output-dir outputs/dynamic_llm_ablation_small`
5. Paper exports
   - `python scripts/hmimo_export_paper_figures.py ... --output-root outputs`
   - `python scripts/hmimo_export_paper_tables.py ... --output-root outputs`

## Expected benchmark outputs
- Static:
  - `hmimo_static_results.json`
  - `hmimo_static_summary.csv`
- Dynamic non-LLM:
  - `hmimo_dynamic_results.json`
  - `hmimo_dynamic_summary.csv`
- Dynamic LLM:
  - `hmimo_dynamic_llm_results.json`
  - `hmimo_dynamic_llm_summary.csv`
- Dynamic LLM ablation:
  - `hmimo_dynamic_llm_ablation_results.json`
  - `hmimo_dynamic_llm_ablation_summary.csv`

## Expected paper export outputs
- Figures (under `outputs/paper_figures`): PNG and PDF for each figure.
- Tables (under `outputs/paper_tables`): compact CSV summaries.

## Seed and config reproducibility
- Every benchmark script reads a YAML config and explicit seeds.
- Monte Carlo or sequence trial variation is generated from deterministic seed offsets.
- Re-running the same command with unchanged config yields the same outputs.
