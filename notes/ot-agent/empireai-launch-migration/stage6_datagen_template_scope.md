# Stage 6 — Datagen template: container-aware srun

**Date:** 2026-07-24
**Status:** scoped GO
**Companion:** README.md (parent)

## Why this is the right next step

Datagen (trace generation via Harbor + Daytona) has never run on EmpireAI. This stage adds the capability using the same container-aware pattern. EmpireAI has full outbound internet, so Daytona sandboxes work — unlike JSC/Leonardo clusters.

## Change-set

### `hpc/datagen_launch_utils.py`

Apply the same container conditional to the `universal_taskgen.sbatch` and `universal_tracegen.sbatch` render paths:
- `srun` gets container flags when `container_image` is set.
- Conda activation is replaced by container env setup.

**Note:** Datagen uses yet another venv layer inside the container — the RL venv (`/opt/envs/rl`) which includes harbor + vllm. The `--container_image` CLI override should point at `mega_v2_rl.sqsh` (or `mega_final_dm.sqsh` if it includes the RL layer — verify).

### Verify the datagen-specific requirements:

- Harbor needs Daytona API access → EmpireAI has outbound internet ✓
- vLLM serve inside the container → verify the RL venv has the correct vLLM fork
- The `--container-mount-home` flag gives access to `$DCFT` (the repo checkout) and `$HF_HOME` ✓

## Validation gate (GO/NO-GO)

1. **Flag-off byte-identical (G1):** Jupiter/Leonardo datagen sbatch unchanged.
2. **Container-on structural check:** EmpireAI datagen `--dry_run` output includes `--container-image` + correct env.
3. **Existing tests:** `pytest tests/hpc/` all pass.

**Cost:** CPU only. (Live datagen smoke is a follow-up — requires a task dataset + running vLLM + Harbor agent loop.)

## Composes with / depends on

Stage 1 + Stage 2 (container fields + methods). Independent of stages 3–5.
