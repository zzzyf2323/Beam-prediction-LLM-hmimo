# Stage 0 Migration Plan (Safe Pass)

This document records the approved **Stage 0** migration approach from the current beam-prediction codebase toward an HMIMO-focused research codebase.

## Scope for this round

- Add planning/inventory/baseline documentation only.
- Add minimal package scaffolding for future HMIMO modules.
- Keep all current beam-prediction runtime behavior unchanged.

## Approved constraints

1. Preserve the HMIMO physical backbone conceptually around FPWS-aligned probing (implementation deferred).
2. Preserve operator-form measurements and adjoint-operator workflows (implementation deferred).
3. Keep Group-SBL and Group-LASSO as target estimators (implementation deferred).
4. Add an optional LLM-assisted temporal prior module in later stages (implementation deferred).
5. Keep LLM components optional in future design.

## Explicit non-goals for Stage 0

- No implementation of FPWS operators.
- No implementation of Group-SBL or Group-LASSO.
- No implementation of LLM temporal prior logic.
- No script move/rename/delete.
- No changes to existing train/test command behavior.

## Safe migration sequence

### Stage 0 (this pass)

- Document migration intent and repository inventory.
- Document baseline beam-prediction commands and I/O artifacts.
- Create empty/near-empty HMIMO package skeleton directories and `__init__.py` files.

### Stage 1 (future)

- Introduce config-first HMIMO experiment entry points alongside legacy scripts.
- Add operator-form interfaces (measurement + adjoint APIs) with smoke tests.
- Add estimator interfaces for Group-SBL and Group-LASSO with placeholders.

### Stage 2 (future)

- Implement FPWS-aligned physics operators in modular `hmimo.physics`.
- Implement Group-SBL and Group-LASSO estimators in `hmimo.estimators`.
- Ensure reproducibility via config + seed controls and baseline scripts.

### Stage 3 (future)

- Add optional LLM temporal prior support tracker in `hmimo.priors`.
- Keep hard toggle to disable LLM and run pure physics + sparse estimators.
- Add smoke/integration tests to validate optionality and no-regression behavior.

## Compatibility promise during migration

Until explicit cutover is approved, existing beam-prediction entry points (`run_main_BP.py`, `test_main_BP.py`, `test_main_BP_neighbor.py`, and shell wrappers under `scripts/`) remain the baseline runtime path.
