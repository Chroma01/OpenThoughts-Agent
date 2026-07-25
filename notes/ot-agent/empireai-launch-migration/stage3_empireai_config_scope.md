# Stage 3 — Complete EmpireAI HPC config

**Date:** 2026-07-24
**Status:** scoped GO
**Companion:** README.md (parent)

## Why this is the right next step

With the container scaffolding in place (stages 1–2), fill in the EmpireAI HPC instance with all fields needed for correct SFT launch. This is where the cluster-specific knowledge from the ops docs and old templates gets encoded into the data model.

## Change-set

**`hpc/hpc.py`** — update the `empireai` HPC instance (currently at line ~1476). Add/modify:

```python
empireai = HPC(
    name="empireai",
    hostname_pattern=r"b\d+-\d+-s\d+-.*",
    dotenv_filename="empireai.env",
    account="ny_chinmayh_datacomp",
    partition="beta",
    gpus_per_node=4,
    cpus_per_node=144,
    internet_node=True,
    gpus_type="B200",
    total_partition_nodes=72,
    gpu_directive_format="--gres=gpu:b200:{n}",
    # --- Container runtime (NEW) ---
    container_image="/mnt/home/bf996/images/mega_final_dm.sqsh",
    container_mount_home=True,
    container_remap_root=False,
    # --- No conda, no modules (env is in the container) ---
    conda_activate="",
    modules=[],
    # --- NCCL + networking (bond0 validated job 31604) ---
    nccl_settings={
        "NCCL_SOCKET_IFNAME": "bond0",
        "GLOO_SOCKET_IFNAME": "bond0",
    },
    # --- Container-internal env vars (PATH sanitize + caches) ---
    env_vars={
        "PATH": "/usr/local/bin:/usr/local/cuda/bin:/usr/bin:/bin",
        "TRITON_CACHE_DIR": "/tmp/triton_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_cache",
        "OMP_NUM_THREADS": "1",
    },
    env_unsets=["PYENV_DIR", "PYENV_ROOT", "PYENV_SHELL"],
    # --- QoS + scheduling ---
    qos="standard",  # ≤36 GPU/48h production default
    default_time_limit="23:59:00",
    max_time_limit="48:00:00",
    num_nodes_default=2,
    num_nodes_slow=1,
    num_nodes_fast=4,
    # --- Training ---
    training_launcher="torchrun",
    ray_tmpdir_base="/tmp/ray",
    # --- Pre-run (page-cache warm for θ₀ mmap) ---
    pre_run_commands=[
        # Warm the page cache for the base model to avoid cold-node mmap faults.
        # The model path is set per-job via exp_args; this is a placeholder.
        # Actual pre-warm happens in SFTJobRunner if configured.
    ],
    # --- Eval listener (already configured, unchanged) ---
    eval_cluster_view={...},  # unchanged from current
)
```

**`hpc/dotenv/empireai.env`** — extend with the missing standard vars:
- `CHECKPOINTS_DIR`, `MODELS_DIR`, `DATASETS_DIR`, `TOKENIZED_DATASETS_DIR`
- `PYTHONPATH` (include `$DCFT`)
- `SCRATCH` (point to `/tmp` or DDN if allocated)

**`hpc/hpc.py:get_sbatch_directives()`** — verify the `--segment` directive is handled. If not currently supported, add an optional `slurm_segment: int | None = None` field that emits `#SBATCH --segment=N` when set. EmpireAI sets `slurm_segment=2` for multi-node jobs.

## Required divergence (documented)

| Item | Old behavior | New behavior | Note |
|------|---|---|---|
| Container image path | `EMPIREAI_IMG` env var, set inline by operator | Hardcoded in HPC instance + overridable via `--container_image` CLI flag | The HPC field is the default; operators can override per-job if they build a new image. |
| QoS selection | `--qos=test/standard/long` passed to `sbatch` manually | `qos="standard"` default + `--qos` CLI override (already supported by `apply_env_overrides`) | No new mechanism needed. |

## Validation gate (GO/NO-GO)

1. **detect_hpc() resolves:** on an EmpireAI login node (hostname matching `b\d+-\d+-s\d+-.*`), `detect_hpc()` returns the `empireai` instance.
2. **get_sbatch_directives() correctness:** renders `#SBATCH --partition=beta`, `#SBATCH --account=ny_chinmayh_datacomp`, `#SBATCH --gres=gpu:b200:N`, `#SBATCH --qos=standard`. Verify against the old `dm_run1_dense.sbatch` header.
3. **get_container_srun_prefix():** returns `--container-image=/mnt/home/bf996/images/mega_final_dm.sqsh --container-mount-home`.
4. **get_nccl_exports():** renders `export NCCL_SOCKET_IFNAME=bond0` + `export GLOO_SOCKET_IFNAME=bond0`.
5. **Flag-off byte-identical (G1):** all non-EmpireAI clusters still render unchanged.

**Cost:** CPU only.

## Composes with / depends on

Stage 1 (container fields) + Stage 2 (container methods on HPC).
