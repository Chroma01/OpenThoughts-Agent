# Stage 5 — RL template: container-aware srun

**Date:** 2026-07-24
**Status:** scoped GO
**Companion:** README.md (parent)

## Why this is the right next step

EmpireAI has an RL venv (`/opt/envs/rl` inside `mega_v2_rl.sqsh` — skyrl_train + vLLM fork + flash-attn) but **no RL sbatch templates at all**. This stage adds RL launch capability using the same container-aware pattern proven in Stage 2.

## Change-set

### `hpc/rl_launch_utils.py`

Apply the same conditional container pattern as Stage 2 did for SFT:
- `srun_prefix` gets `hpc.get_container_srun_prefix()` appended when `container_image` is set.
- The RL-specific env setup uses `hpc.get_container_env_setup()` instead of conda activation.

**Key difference from SFT:** EmpireAI RL uses a **different image** (`mega_v2_rl.sqsh`) than SFT (`mega_final_dm.sqsh`). This requires either:
- (a) A separate HPC field `container_image_rl: str | None = None` (clean but adds a field), or
- (b) An `--container_image` CLI override that the RL launcher reads (simpler, per-job flexibility).

**Recommendation:** Option (b) — `--container_image` CLI override, defaulting to the HPC instance's `container_image`. This lets SFT and RL use different images without duplicating the field. The RL launcher passes `exp_args.get("container_image") or hpc.container_image`.

### `hpc/hpc.py`

If option (b): no new field needed — `container_image` on the HPC instance is the SFT default; RL overrides via CLI.

### `hpc/sbatch_rl/universal_rl.sbatch`

Verify it has the same `{conda_activate}`, `{module_commands}`, `{srun_command}` substitution slots as the SFT template. If the RL template has a different structure, adapt the container conditional accordingly.

## Required divergence (documented)

| Item | Note |
|------|------|
| **RL image ≠ SFT image** | RL uses `/opt/envs/rl` venv inside `mega_v2_rl.sqsh` (or the combined `mega_final_dm.sqsh` which includes the RL layer). Verify which image has the RL env and set the CLI override accordingly. |
| **No existing RL templates** | Unlike SFT (which has `dm_run1_dense.sbatch` as reference), RL has no old EmpireAI template to diff against. The gate is structural correctness + a live smoke. |

## Validation gate (GO/NO-GO)

1. **Flag-off byte-identical (G1):** Jupiter/Leonardo RL sbatch unchanged.
2. **Container-on structural check:** EmpireAI RL `--dry_run` output includes `--container-image` + bond0 NCCL + PATH sanitize.
3. **Existing tests:** `pytest tests/hpc/` all pass.

**Cost:** CPU only. (Live RL smoke is a follow-up — RL requires multi-node + Daytona sandboxes, which is a campaign-scale validation, not a stage gate.)

## Composes with / depends on

Stage 1 + Stage 2 (container fields + methods). Independent of Stage 3–4 (SFT config), but benefits from the EmpireAI HPC config being complete.
