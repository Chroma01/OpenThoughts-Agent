# Stage 4 — SFT end-to-end gate

**Date:** 2026-07-24
**Status:** scoped GO
**Companion:** README.md (parent)

## Why this is the right next step

This is the parity gate — prove that `python -m hpc.launch --job_type sft` on EmpireAI produces correct, runnable output. If this passes, the container-aware template machinery is proven and stages 5–6 (RL, datagen) are mechanical application of the same pattern.

## Change-set

No new code this stage — this is the validation gate for stages 1–3. However, minor fixes discovered during the diff may be applied here (e.g. a missing env var, a wrong directive order).

## Validation gate (GO/NO-GO)

### Gate A — Dry-run structural parity (CPU, on login node)

1. Run `python -m hpc.launch --job_type sft --dry_run <standard args>` on an EmpireAI login node.
2. Read the generated `<job>_sft.sbatch`.
3. Diff against `hpc/empireai/dm_run1_dense.sbatch` (the reference).
4. **Required matches:**
   - `#SBATCH --gres=gpu:b200:N` ✓
   - `#SBATCH --partition=beta` ✓
   - `#SBATCH --account=ny_chinmayh_datacomp` ✓
   - `#SBATCH --qos=standard` ✓
   - `srun ... --container-image=mega_final_dm.sqsh --container-mount-home` ✓
   - `export NCCL_SOCKET_IFNAME=bond0` ✓
   - `export GLOO_SOCKET_IFNAME=bond0` ✓
   - PATH sanitize (`/usr/local/bin:...`) ✓
   - Node-local cache dirs (`/tmp/triton_cache`, etc.) ✓
5. **Expected + acceptable differences:**
   - Training invocation: `python -m hpc.sft_launch_utils --config <json>` instead of inline `axolotl train` — **intentional** (the Python runner IS the migration).
   - Template structure differs (the universal template has the write-cache guard, the WORKDIR sanity check, etc. that the old template lacks) — **intentional** (these are improvements the universal template provides for free).

### Gate B — Live smoke (optional, 2 GPU, ~10 min)

If Gate A passes and the operator approves:
1. Submit a tiny SFT smoke: `python -m hpc.launch --job_type sft --model Qwen/Qwen2.5-0.5B --dataset <tiny> --max_steps 2 --num_nodes 1 --qos test`
2. Verify: container loads, torch sees B200 GPUs (`torch.cuda.device_count() == 4`), training runs 2 steps, checkpoint saved.
3. This is the equivalent of `sft_axolotl_smoke.sbatch` but via `hpc.launch`.

### Gate C — Existing cluster regression

`pytest tests/hpc/` all pass. Additionally, `--dry_run` on Jupiter/Leonardo still produces byte-identical sbatch output (re-verify G1 after stages 1–3 are all landed).

**Cost:** Gate A = CPU (login node). Gate B = 2 GPU × 10 min (optional). Gate C = CPU.

## Composes with / depends on

Stages 1 + 2 + 3 all landed and green.
