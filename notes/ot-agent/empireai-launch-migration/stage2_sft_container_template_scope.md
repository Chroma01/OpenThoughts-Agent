# Stage 2 — SFT template: container-aware srun

**Date:** 2026-07-24
**Status:** scoped GO
**Companion:** README.md (parent)

## Why this is the right next step

SFT is the proven EmpireAI workload (DenseMixer ablation). Making the SFT universal template container-aware is the critical path — once this works, RL and datagen follow the same pattern (stages 5–6).

## Change-set

### `hpc/hpc.py` — add two methods to HPC:

```python
def get_container_srun_prefix(self) -> str:
    """Return srun container flags, or empty string if no container."""
    if self.container_image is None:
        return ""
    flags = [f"--container-image={self.container_image}"]
    if self.container_mount_home:
        flags.append("--container-mount-home")
    if self.container_remap_root:
        flags.append("--container-remap-root")
    if self.container_extra_args:
        flags.append(self.container_extra_args)
    return " ".join(flags)

def get_container_env_setup(self) -> str:
    """Return container-internal env-setup exports, or empty string.
    For EmpireAI: PATH sanitize (scrub host pyenv shims) + bond0 NCCL."""
    if self.container_image is None:
        return ""
    lines = []
    for key, val in self.env_vars.items():
        lines.append(f'export {key}="{val}"')
    return "\n".join(lines) if lines else ""
```

### `hpc/sft_launch_utils.py` — modify `construct_sft_sbatch_script` (line ~1142):

The `srun_prefix` and `conda_activate` become conditional:

```python
# Current (line 1142):
srun_prefix = f"srun --nodes={num_nodes} --ntasks-per-node=1"

# New:
srun_prefix = f"srun --nodes={num_nodes} --ntasks-per-node=1"
container_flags = hpc.get_container_srun_prefix()
if container_flags:
    srun_prefix += f" {container_flags}"
```

And the `conda_activate` substitution (line ~1156–1167):
```python
# Current:
conda_activate = _get_sft_conda_activate(hpc, exp_args)
cmd = f'{conda_activate} && python -m hpc.sft_launch_utils --config "{config_path}"'

# New:
if hpc.container_image is not None:
    # Container path: no conda, the env is inside the .sqsh.
    # Use the container_env_setup (PATH sanitize, NCCL bond0, etc.) instead.
    container_setup = hpc.get_container_env_setup()
    cmd = f'{container_setup}; python -m hpc.sft_launch_utils --config "{config_path}"'
else:
    conda_activate = _get_sft_conda_activate(hpc, exp_args)
    cmd = f'{conda_activate} && python -m hpc.sft_launch_utils --config "{config_path}"'
```

Also: the `{conda_activate}` template substitution (line 1167) should be empty string for containerized clusters (the env is in the container, not in the sbatch preamble). And the `{module_commands}` substitution should also be empty for EmpireAI (no `module load` — CUDA is in the container).

### `hpc/sbatch_sft/universal_sft.sbatch` — NO change needed

The template already has `{conda_activate}` and `{module_commands}` as substitution slots. When those resolve to empty strings, the template is correct for containerized clusters. The `{srun_command}` slot carries the container flags. **This is the elegance of the existing design** — no template edit needed, only the substitution values change.

## Required divergence (documented)

| Item | Old `dm_run1_dense.sbatch` | New `hpc.launch` path | Divergence reason |
|------|---|---|---|
| Training invocation | Inline `axolotl train --config ...` | `python -m hpc.sft_launch_utils --config <json>` | **Intentional** — the Python runner is the unified path; it will call axolotl (or LLaMA-Factory) internally via the SFTJobRunner. This IS the migration. |
| DenseMixer patch | `densemixer_qwen3moe_tf5.py` copied over the pip file in the image | Same — the image already has the patch; no launcher-side change | The patch lives in the container, not the launcher. |
| Page-cache pre-warm | Explicit `cat θ₀ > /dev/null &` in the old template | Add as a `pre_run_commands` entry on the HPC instance | Cleaner — `pre_run_commands` is the existing extension point. |

## Validation gate (GO/NO-GO)

1. **Flag-off byte-identical (G1):** render `universal_sft.sbatch` for Jupiter (no container_image) before and after the code change. Zero-byte diff. Unit test: `test_container_flag_off_identical`.
2. **Container-on structural match:** render for a synthetic test cluster with `container_image="/fake/test.sqsh"`. Verify:
   - `{srun_command}` contains `--container-image=/fake/test.sqsh --container-mount-home`
   - `{conda_activate}` resolves to empty string
   - `{module_commands}` resolves to empty string
3. **Existing tests:** `pytest tests/hpc/` all pass.

**Cost:** CPU only.

## Composes with / depends on

Stage 1 (the `container_image` field must exist on HPC).
