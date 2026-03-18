# HMIMO results inventory

## Benchmark result files

### Static benchmark
- `hmimo_static_results.json`: per-trial raw records + summary.
- `hmimo_static_summary.csv`: aggregated metrics by scheme/SNR.

### Dynamic non-LLM benchmark
- `hmimo_dynamic_results.json`: per-frame/per-trial records + summary.
- `hmimo_dynamic_summary.csv`: aggregated dynamic metrics.

### Dynamic LLM benchmark
- `hmimo_dynamic_llm_results.json`: per-frame/per-trial records + summary for no-temporal/non-LLM/LLM variants.
- `hmimo_dynamic_llm_summary.csv`: aggregated dynamic LLM benchmark metrics.

### Dynamic LLM ablation benchmark
- `hmimo_dynamic_llm_ablation_results.json`: raw records for all ablation variants.
- `hmimo_dynamic_llm_ablation_summary.csv`: aggregated ablation metrics.

## Paper export files

### Figures (`outputs/paper_figures`)
- `fig_static_nmse_vs_snr.png` / `.pdf`
- `fig_dynamic_temporal_compare.png` / `.pdf`
- `fig_dynamic_llm_ablation.png` / `.pdf`

### Tables (`outputs/paper_tables`)
- `table_static_summary.csv`
- `table_dynamic_summary.csv`
- `table_dynamic_llm_summary.csv`
- `table_dynamic_llm_ablation_summary.csv`
