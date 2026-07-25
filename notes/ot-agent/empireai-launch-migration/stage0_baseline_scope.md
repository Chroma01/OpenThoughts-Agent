# Stage 0 — Baseline + evidence

**Date:** 2026-07-24
**Status:** scoped GO
**Companion:** README.md (parent)
**No code changes this stage — docs/evidence only.**

## Why this is the right next step

Before any code, capture the exact behavior of the existing EmpireAI parallel path so the parity gate (Stage 4) has a concrete reference. This stage produces the "golden" outputs and behavioral spec that later stages are validated against.

## Change-set

No `<repo>/` source touched. Create:
- `agent_logs/2026-07-24_empireai-migration-scoping.md` — the evidence log

Capture into the log:
1. **The `dm_run1_dense.sbatch` anatomy** — annotate every line: which are standard SLURM, which are EmpireAI-specific (QoS, gres, segment), which are container-specific (`--container-image`, `--container-mount-home`, `--container-remap-root`), which are workload-specific (axolotl, ZeRO-3, DenseMixer).
2. **The container image resolution** — how `mega_final_dm.sqsh` is located (path on the cluster), how the `EMPIREAI_IMG` env var is set.
3. **The env-var ritual** — every export inside the container: `NCCL_SOCKET_IFNAME=bond0`, `GLOO_SOCKET_IFNAME=bond0`, `PATH=/usr/local/bin:...` (pyenv scrub), `OMP_NUM_THREADS`, node-local cache dirs (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`), `HF_HOME`, `WANDB_*`.
4. **The QoS tier mapping** — test (≤4 GPU/6h), standard (≤36 GPU/48h), long (≤7d), priority (2×/24h). Which tier does a production SFT run use?
5. **The `--segment` directive** — validated at `--segment=2` for 2-node jobs. When is it needed?
6. **The existing eval integration** — verify `python -m hpc.launch --job_type eval_listener` works on EmpireAI today (it should — the eval path is already containerized via `eval/empireai/eval_harbor.sbatch`).

## Validation gate (GO/NO-GO)

Scoping log written with all 6 items above. The `dm_run1_dense.sbatch` annotation is complete enough that a reader could reconstruct the sbatch from it.

## Composes with / depends on

Nothing — this is the starting point.
