# EmpireAI → hpc.launch Migration Plan

**Date:** 2026-07-24
**Status:** scoped — propose-only; no code yet
**Target repo:** `OpenThoughts-Agent` (canonical local path: `~/Documents/OpenThoughts-Agent`)
**Isolated working copy:** `~/Documents/staged-work/empireai-migration/OpenThoughts-Agent`
**Branch:** `feuer/empireai-launch-migration`
**Evidence:** agent_logs/2026-07-24_empireai-migration-scoping.md

## Goal

Fold the EmpireAI Beta parallel path (`hpc/empireai/` standalone sbatch + Pyxis/Enroot containers) into the unified `python -m hpc.launch` entrypoint. After this change, `python -m hpc.launch --job_type {sft,rl,datagen}` works on EmpireAI with identical behavior to the existing hand-authored templates. The old `hpc/empireai/` files are retained for reproducibility.

**Testable end state:**
1. `python -m hpc.launch --job_type sft` on an EmpireAI login node produces an sbatch script byte-equivalent (modulo paths/job-IDs) to the hand-authored `dm_run1_dense.sbatch`, using the same container image + same env vars + same srun flags.
2. All existing clusters produce **byte-identical** sbatch output when their `container_image` field is unset (the flag-off invariant).
3. The old `hpc/empireai/*.sbatch` files remain functional standalone.

## Why / the mechanism

`hpc.launch` generates sbatch scripts by substituting `{placeholder}` values from the `HPC` pydantic model into universal templates. The universal templates assume **conda env activation** — both the `{conda_activate}` slot (line 51 of `universal_sft.sbatch`) and the `{srun_command}` (which wraps `srun ... bash -c '<conda activate> && python ...'`).

EmpireAI uses **Pyxis/Enroot containers** instead of conda. The env lives inside a `.sqsh` image (3 layered venvs: SFT/axolotl, RL/skyrl, JAX/levanter). The `srun` invocation needs `--container-image=<sqsh> --container-mount-home`, and the conda-activate slot should be a no-op (or a PATH-sanitize block to scrub the host's x86 pyenv shims that leak via `--container-mount-home`).

The fix is a **conditional container runtime** threaded through the HPC model → the template substitution → the srun command. When `container_image` is unset (all existing clusters), every code path is unchanged.

## Stage map

| Stage | Title | What | Layer | Cost | Gate |
|-------|-------|------|-------|------|------|
| 0 | Baseline + evidence | Document exact EmpireAI sbatch behavior; verify existing eval integration; capture the byte-level reference output of `dm_run1_dense.sbatch` | docs | CPU | scoping log written |
| 1 | HPC container fields | Add `container_image`, `container_mounts`, `container_remap_root`, `container_extra_args` to HPC model (all Optional/default-off) | `hpc/hpc.py` | CPU | flag-off: existing tests pass; `empireai.container_image` resolves to the mega sqsh |
| 2 | SFT template: container-aware srun | Thread container fields through `construct_sft_sbatch_script` → `{srun_command}` + `{conda_activate}`. Add `get_container_srun_prefix()` + `get_container_env_setup()` methods to HPC | `hpc/hpc.py`, `hpc/sft_launch_utils.py` | CPU | flag-off byte-identical on Jupiter/Leonardo; container-aware output matches `dm_run1_dense.sbatch` structure |
| 3 | EmpireAI HPC config completion | Fill in all SFT-relevant fields: QoS tiers, `env_vars` (bond0 NCCL, PATH sanitize), `conda_activate=""`, `modules=[]`, QoS dispatch via `extra_sbatch_directives`, segment directive | `hpc/hpc.py` | CPU | `detect_hpc()` resolves on EmpireAI hostname; `get_sbatch_directives()` emits correct `--gres`/`--qos`/`--segment` |
| 4 | SFT end-to-end gate | Launch `python -m hpc.launch --job_type sft --dry_run` targeting EmpireAI; diff generated sbatch vs `dm_run1_dense.sbatch`; resolve discrepancies; optional live 2-GPU smoke | cluster | 1-GPU (optional) | `--dry_run` output structurally identical to old template; live smoke trains + checkpoints |
| 5 | RL template: container-aware srun | Same container-threading for `universal_rl.sbatch` + `rl_launch_utils`. EmpireAI RL config (the `/opt/envs/rl` venv inside `mega_v2_rl.sqsh`) | `hpc/hpc.py`, `hpc/rl_launch_utils.py` | CPU | flag-off byte-identical; EmpireAI RL sbatch matches expected container structure |
| 6 | Datagen template: container-aware srun | Same for `universal_taskgen.sbatch` / `universal_tracegen.sbatch`. EmpireAI datagen config | `hpc/hpc.py`, `hpc/datagen_launch_utils.py` | CPU | flag-off byte-identical; EmpireAI datagen sbatch structurally correct |
| 7 | Retain old artifacts | Mark `hpc/empireai/` as retained-for-reproducibility; add `hpc/empireai/DEPRECATED.md` pointing to `hpc.launch`; update `hpc/README.md` supported-cluster table | docs | CPU | old templates still run standalone; README updated |

**Critical path:** 0 → 1 → 2 → 3 → 4 (SFT is the proven workload). Stages 5–6 are independent of each other and build on 1+2's container scaffolding.

## Global invariants

**G1 — Flag-off byte-identical (THE gating invariant):**
When `container_image` is unset (None/absent — the default for ALL existing clusters), the sbatch output of every universal template MUST be byte-identical to the current output. This is asserted by a unit test that renders the template for a sentinel cluster (e.g. Jupiter) before and after each stage, and diffs the output. The test fails on any byte difference.

**G2 — Parity with old EmpireAI templates:**
The `--dry_run` sbatch output for EmpireAI SFT must match the structure of `dm_run1_dense.sbatch` — same container image, same env vars (bond0, PATH sanitize, node-local caches), same QoS/gres/segment directives, same srun container flags. Differences must be documented and justified (e.g. the Python runner replaces inline axolotl invocation — that IS the point of the migration).

**G3 — Minimal diff:**
No gratuitous changes to existing clusters' HPC instances, the universal templates' non-container code paths, or the `sft_launch_utils` / `rl_launch_utils` render functions beyond the container conditional. The diff should be: new HPC fields + new methods + conditional branches + EmpireAI config completion.

**G4 — Old templates remain functional:**
The `hpc/empireai/*.sbatch` files are NOT deleted, NOT modified, NOT moved. They remain valid standalone sbatch scripts. A `DEPRECATED.md` is added pointing to the `hpc.launch` path as canonical.

## Borrow map (anchors — RECONFIRM at impl time, they drift)

| What | File | Anchor (from 2026-07-24 read) |
|------|------|------|
| HPC model definition | `hpc/hpc.py:10` | `class HPC(BaseModel)` with ~50 fields |
| `empireai` HPC instance | `hpc/hpc.py:1476` | partial registration (eval only) |
| `clusters` list | `hpc/hpc.py:1773` | append `empireai` already present |
| `detect_hpc()` | `hpc/hpc.py:1776` | hostname regex match loop |
| `set_environment()` | `hpc/hpc.py:1797` | dotenv parser |
| `get_sbatch_directives()` | `hpc/hpc.py:288` | emits `#SBATCH -p/--gres/...` |
| `get_module_commands()` | `hpc/hpc.py` | renders `module load` lines |
| `get_env_exports()` | `hpc/hpc.py` | renders `export KEY=VAL` lines |
| `get_nccl_exports()` | `hpc/hpc.py` | renders NCCL env exports |
| SFT srun_command construction | `hpc/sft_launch_utils.py:1142-1158` | `srun_prefix` + `conda_activate` + `cmd` |
| SFT substitutions dict | `hpc/sft_launch_utils.py:1159-1180` | the `{placeholder}` → value map |
| `universal_sft.sbatch` template | `hpc/sbatch_sft/universal_sft.sbatch` | 220 lines, conda at :51, srun at :220 |
| `_get_sft_conda_activate()` | `hpc/sft_launch_utils.py` | returns the conda-activate string |
| RL launch utils | `hpc/rl_launch_utils.py` | `launch_rl_job(exp_args, hpc)` |
| `universal_rl.sbatch` | `hpc/sbatch_rl/universal_rl.sbatch` | RL template |
| Datagen launch utils | `hpc/datagen_launch_utils.py` | `launch_datagen_job_v2(exp_args, hpc)` |
| `universal_taskgen.sbatch` | `hpc/sbatch_data/universal_taskgen.sbatch` | task extraction template |
| `universal_tracegen.sbatch` | `hpc/sbatch_data/universal_tracegen.sbatch` | trace generation template |
| EmpireAI dotenv | `hpc/dotenv/empireai.env` | 12 lines, minimal |
| EmpireAI old templates | `hpc/empireai/*.sbatch` | dm_run1_dense.sbatch etc. |
| EmpireAI ops docs | `.claude/ops/empireai/ops.md` | cluster particulars |

## Safety / reward-hacking

N/A — this is a launcher refactor, not RL reward shaping. The only safety concern is operational: a broken `hpc.launch` integration could silently submit jobs with wrong container flags on EmpireAI (wasted SU budget). The `--dry_run` gate (Stage 4) catches this before any real submission.

## Validation discipline

Per stage:
1. **Flag-off byte-identical FIRST** — render the universal template for a non-container cluster (Jupiter), diff against pre-change output. Zero diff = pass.
2. **Container-on structural match** — render for EmpireAI, verify the container/srun/env structure matches the old hand-authored template.
3. **Existing test suite** — `pytest tests/hpc/` must remain green.
4. **Live smoke (optional, Stage 4 only)** — submit a tiny 2-GPU SFT job on EmpireAI via `hpc.launch`, verify it enters training.
