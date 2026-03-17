# Baseline Beam-Prediction Commands (Pre-Migration)

This document captures the **current** train/test entry points and their expected inputs/outputs before HMIMO migration changes.

## Environment baseline

- Install Python dependencies:

```bash
pip install -r requirements.txt
```

- Dataset generation workflow (as documented):
  1. Install `DeepMIMO`.
  2. Place scenario data under `deepmimo_generate_data/scenarios`.
  3. Run `deepmimo_generate_data/gen_training_data.py`.
  4. Place generated data under `dataset/`.

## Training entry points

### A) Canonical script launcher

```bash
bash ./scripts/LLM_BP.sh
```

What it currently does:

- Launches `run_main_BP.py` via `accelerate launch`.
- Uses `--root_path ./dataset/BP_dataset/`.
- Uses `--checkpoints ./checkpoints/`.
- Trains with default script loop values (single-value arrays by default).

### B) Direct Python launcher equivalent (representative)

```bash
accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 10097 run_main_BP.py \
  --M_phi 0 \
  --checkpoints ./checkpoints/ \
  --speeds 10 15 20 \
  --num_antenna 32 64 128 \
  --patience 5 \
  --islora 0 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/BP_dataset/ \
  --model_id BP_32_16 \
  --model TimeLLM \
  --data BP \
  --features MS \
  --seq_len 40 \
  --pred_len 10 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --patch_len 1 \
  --stride 1 \
  --des Exp \
  --itr 1 \
  --num_tokens 200 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --llm_layers 32 \
  --train_epochs 50 \
  --model_comment TimeLLM-BP \
  --load_model 0 \
  --llm_model GPT2 \
  --llm_dim 1280
```

Expected training inputs:

- Dataset tree rooted at `./dataset/BP_dataset/`.
- Optional model assets (e.g., GPT2-local directory if configured by model code).

Expected training outputs:

- Checkpoint directory under `./checkpoints/` using an auto-generated experiment `setting` name.
- Best checkpoint file named `checkpoint` inside the experiment directory (loaded by test scripts when `--load_model 1`).

## Testing entry points

### A) Canonical script launcher

```bash
bash ./scripts/LLM_BP_test.sh
```

What it currently does:

- Launches `test_main_BP.py` via `accelerate launch`.
- Uses `--is_training 0` and `--load_model 1`.
- Reads checkpoints from `./checkpoints`.

### B) Alternate neighbor-based test launcher

```bash
accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 10099 test_main_BP_neighbor.py <args...>
```

Used for neighbor-assisted beam index post-processing/evaluation.

Expected testing inputs:

- Dataset tree rooted at `./dataset/BP_dataset/`.
- Trained checkpoint(s) under `./checkpoints/` matching the generated experiment setting.

Expected testing outputs:

- Result directory created at `./dataset/BP_dataset/result`.
- CSV artifacts from `test_main_BP.py`, including:
  - `beam_label_pred_*.csv`
  - `beam_label_true_*.csv`
  - `beam_label_gain_*.csv`
- CSV artifacts from `test_main_BP_neighbor.py`, including:
  - `beam_label_pred_*.csv`
  - `beam_label_gain_*.csv`

## Stage 0 safety note

These commands are documented only; no behavioral changes are introduced in Stage 0.
