# Project rules for Codex

## Goal
This repository is being adapted from an LLM-based beam prediction codebase into an HMIMO research codebase.

## What Codex must do
1. Keep the HMIMO physical backbone based on FPWS-aligned probing.
2. Keep operator-form measurement and adjoint operators.
3. Keep Group-SBL and Group-LASSO as the main estimators.
4. Add an LLM-assisted temporal prior module for dynamic wavenumber-domain support tracking.
5. Make the LLM optional.

## What Codex must not do
1. Do not replace the HMIMO physical solver with an end-to-end LLM.
2. Do not delete original license files.
3. Do not collapse the whole project into one script.
4. Do not build giant explicit sensing matrices in the main HMIMO path unless only for tiny debugging.
5. Do not change the scientific target back into beam prediction.

## Engineering rules
1. Keep the code modular.
2. Separate data, models, hmimo, scripts, tests, and docs.
3. Add docstrings to public functions.
4. Every important new module should have at least a smoke test.
5. All runs must be reproducible by config and seed.

## SBL reference authority

The authoritative reference implementation for HMIMO Group-SBL is:

- reference_impl/demo_sbl_block_operator_fixed_angles_report.py
- reference_impl/demo_compare_beamforming_mechanisms_dft_strictAligned.py

When modifying HMIMO SBL-related code, treat the above files as the source of truth.

Hard rules:
1. Do not invent a new SBL update rule.
2. Do not simplify away the Hutchinson-style diagonal correction unless explicitly asked.
3. Preserve the same vec(Ha) convention as the reference implementation.
4. Preserve the same group indexing convention as the reference implementation.
5. Preserve the same forward/adjoint operator semantics as the reference implementation.
6. Before editing code, first compare the current repo implementation against the reference files and explain all mismatches.
7. After editing, run tests and also run a numerical consistency comparison against the reference implementation where feasible.

Definition of done for SBL work:
1. Current repo SBL path matches the reference implementation logic.
2. Static HMIMO benchmark trend is no longer obviously broken.
3. No end-to-end LLM estimator is introduced.
4. Group-SBL remains the final estimator, and the LLM remains prior-only.
