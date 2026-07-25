# Stage 1 — Add container runtime fields to HPC model

**Date:** 2026-07-24
**Status:** scoped GO
**Companion:** README.md (parent)

## Why this is the right next step

The HPC model needs new optional fields before any template/render code can branch on them. This stage is pure data-model extension — no rendering logic changes, no template edits. It's the foundation stages 2–6 build on.

## Change-set

**`hpc/hpc.py`** — add 4 new fields to the `HPC` class:

```python
# --- Container runtime (Pyxis/Enroot) ---
# When set, the universal sbatch templates use `srun --container-image=<path>`
# instead of conda activation. All existing clusters leave this unset (None) →
# byte-identical conda-based behavior (flag-off invariant G1).
container_image: str | None = None          # path to .sqsh (e.g. "/mnt/home/bf996/images/mega_final_dm.sqsh")
container_mount_home: bool = True           # --container-mount-home
container_remap_root: bool = False          # --container-remap-root
container_extra_args: str = ""              # additional srun --container-* flags (e.g. "--container-mounts=...")
```

No other fields change. No methods added yet (that's Stage 2). No existing cluster instance is modified.

## Validation gate (GO/NO-GO)

1. **Flag-off byte-identical (G1):** `pytest tests/hpc/` — all existing tests pass unchanged.
2. **Field defaults:** verify `jupiter.container_image is None`, `leonardo.container_image is None`, etc. for every cluster in the `clusters` list. A quick unit assertion: `[c.name for c in clusters if c.container_image is not None] == []`.
3. **No template impact:** the universal sbatch templates are NOT modified this stage — verify by rendering a test sbatch for any cluster and confirming it's unchanged.

**Cost:** CPU only (no GPU, no cluster access needed).

## Composes with / depends on

Stage 0 (evidence log — confirms what container_image path / mount behavior to set later).
