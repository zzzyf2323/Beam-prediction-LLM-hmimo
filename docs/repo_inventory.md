# Repository Inventory (Stage 0)

This inventory classifies current repository components into three buckets to support a safe HMIMO migration.

## 1) Reusable files/modules

These are likely reusable with moderate adaptation:

- `utils/`
  - Utility helpers (training utilities, scheduling, early stopping, content loading patterns).
- `data_provider/`
  - Data loading abstraction and split-based provider pattern.
- `layers/`
  - Reusable neural building blocks that may be retained for optional learning-assisted modules.
- `models/`
  - Reference model structures useful as legacy baselines and for optional LLM-assisted prior modules.
- `requirements.txt`
  - Existing dependency baseline to preserve reproducibility for legacy scripts.
- `ds_config_zero2.json`
  - Existing distributed/deepspeed configuration reference.

## 2) Beam-specific files/modules

These are the active beam-prediction runtime paths and should remain unchanged during Stage 0:

- `run_main_BP.py`
  - Main training/evaluation path for beam prediction.
- `test_main_BP.py`
  - Main test/inference path for beam prediction.
- `test_main_BP_neighbor.py`
  - Neighbor-based beam test/inference variant.
- `scripts/LLM_BP.sh`
  - Canonical training launcher for beam prediction.
- `scripts/LLM_BP_test.sh`
  - Canonical testing launcher for beam prediction.
- `dataset/BP_dataset/` (expected runtime location)
  - Beam-prediction input data root used by scripts.

## 3) Legacy files to isolate later

These are legacy assets that should be isolated behind a `legacy` compatibility boundary in later stages (without deleting now):

- Beam-specific top-level entry points:
  - `run_main_BP.py`
  - `test_main_BP.py`
  - `test_main_BP_neighbor.py`
- Beam-specific shell wrappers:
  - `scripts/LLM_BP.sh`
  - `scripts/LLM_BP_test.sh`
- Beam-oriented dataset generation assets:
  - `deepmimo_generate_data/`
  - `dataset/prompt_bank/`

## Notes

- No files are moved, renamed, or deleted in Stage 0.
- HMIMO package scaffolding is added in parallel and does not alter current runtime paths.
