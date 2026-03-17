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
