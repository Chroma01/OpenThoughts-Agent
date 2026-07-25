# Stage 7 — Retain old artifacts + update docs

**Date:** 2026-07-24
**Status:** scoped GO
**Companion:** README.md (parent)

## Why this is the right next step

Close out the migration: mark the old parallel path as superseded, update the docs, and ensure reproducibility of past experiments.

## Change-set

### `hpc/empireai/DEPRECATED.md` (NEW)

A short notice:
```
# EmpireAI — Legacy sbatch templates

These templates are the original EmpireAI Beta bring-up + DenseMixer ablation
artifacts. They remain functional for standalone use and experiment
reproducibility.

For new launches, use the unified launcher:
  python -m hpc.launch --job_type {sft,rl,datagen}

See notes/ot-agent/empireai-launch-migration/ for the migration plan.
```

### `hpc/hpc.py` — update the EmpireAI comment block

Change the NOTE at line ~1468 from "NOT yet fully `python -m hpc.launch`-integrated" to reflect the new integrated status.

### `hpc/README.md` — update the supported-cluster table

Add/update the EmpireAI row to show SFT/RL/datagen/eval as supported via `hpc.launch`.

### `hpc/empireai/` files — NOT modified

All `.sbatch`, `.sh`, `.py`, `Dockerfile.*` files remain untouched. They are the reproducibility artifacts for the DenseMixer experiment (Run 1 dense / Run 2 sparse) and the container-build chain.

## Validation gate (GO/NO-GO)

1. Old templates still parse: `sbatch --test hpc/empireai/dm_run1_dense.sbatch` does not error (if on cluster, or verify syntax locally).
2. `DEPRECATED.md` exists and points to `hpc.launch`.
3. README table updated.
4. `pytest tests/hpc/` all pass.

**Cost:** CPU only.

## Composes with / depends on

All prior stages landed (1–6).
