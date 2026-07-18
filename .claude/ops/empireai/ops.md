# Empire AI — cluster ops (Alpha H100 + Beta B200/NVL72)

Empire AI is a NY-state consortium cluster. **Two distinct systems**, different login hosts + CPU arch:

| System | Login (ssh alias) | Login arch | Compute | Container model |
|---|---|---|---|---|
| **Alpha** (H100) | `EmpireAI_Alpha1` → `alpha1.empire-ai.org` | **x86_64** (Rocky 9.4) | `alphagpu01-24` = 8× **H100 80GB** on x86 Xeon; per-university partitions (`nyu`, columbia, …); GPUs=gres; + `grace` partition = **Grace-Grace CPU-only aarch64** (`betagg01-60`, 144-core, 490GB, NO GPU) + a down GH200 | venv/conda OR containers |
| **Beta** (B200) ← **the one we want** | `EmpireAI_Beta` → `beta.empireai.edu` | **aarch64** (Ubuntu 24.04) | **NVL72 SuperPOD** — see below | **Pyxis/Enroot MANDATORY** |

> ⚠ Alpha and Beta are **separate logins** (different domains: `empire-ai.org` vs `empireai.edu`, different IPs). The `beta` partition does NOT exist on the alpha1 login. Do not conflate them.

## Access — 2FA + ControlMaster (I ride the operator's socket)

- **Login is 2FA keyboard-interactive**: `(user@host) Authenticator code:` THEN `(user@host) Password:`. The plain `password` method is **rejected** — the ssh config must set `PreferredAuthentications keyboard-interactive,password` + `PubkeyAuthentication no` + `IdentitiesOnly yes` (else "Too many authentication failures"). Both `EmpireAI_Alpha1` and `EmpireAI_Beta` blocks in `~/.ssh/config` are set this way, with `ControlMaster auto` + `ControlPath ~/.ssh/sockets/%r@%h-%p` + `ControlPersist 8h`.
- **I cannot do the 2FA** (no TTY, no authenticator). The operator runs `ssh EmpireAI_Beta` **in a real interactive terminal** (NOT the `!` prefix — no TTY there → it fails) to establish the master socket; then I reuse it non-interactively (`ssh -o BatchMode=yes EmpireAI_Beta …`) for ~8h. When the socket idles out, ask the operator to reconnect.
- User = `bf996` (netid); groups: `bf996`, **`nyu`**.

## Beta hardware — the B200 NVL72

- **`beta*` partition** (default). **72 DGX nodes**, each `Gres=gpu:b200:4` → **4× NVIDIA B200 per node = 288 B200 total** (4 NVL72 racks). Node names `bN-NN-sN-dgx-01-cNN`.
- Per node: **144 aarch64 Grace cores** (2× 72-core sockets), **~1.5 TB RAM** (1561507 MB). GPUs **ARE** SLURM gres here (unlike TACC / the alpha grace-CPU nodes) → request `--gres=gpu:b200:N` or `--gpus-per-node=4`.
- **B200 = Blackwell = `sm_100`** → containers need **CUDA ≥ 12.8** + recent PyTorch/vLLM/Axolotl/Marin builds. (Confirm `compute_cap` in-container on first smoke test.)
- Capacity is real: 64/72 idle at first survey. Doc: use **`--segment {1,4,8,16}`** to control multi-node placement locality (matters for MoE all-to-all).

## Networking — compute-node egress (validated)

**Beta compute nodes have FULL outbound internet** (validated 2026-07-18 from `b1-36-s1-dgx-01` via a qos=test srun: `app.daytona.io`=200, `api.daytona.io`=301, `a.pinggy.io:443` OPEN, `huggingface.co`=200, `www.google.com`=200). **No proxy, no SOCKS, no step-ca cert** — contrast Leonardo (airgapped compute → agentic-opencode/Daytona infeasible there). So from a Beta compute node: **agentic evals work** (Daytona sandboxes + the outbound pinggy SSH tunnel + HTTPS to the Daytona REST API), and **online HF pulls work**.

- The SFT running with `HF_HUB_OFFLINE=1` is a **chosen prestage** ("θ₀+data all local", per the sbatch's own comment), **NOT** a network constraint — proven by `huggingface.co=200` from a compute node.
- (The login/mgmt node is minimal; egress is a compute-node property, tested via srun.)

## SLURM — 25.05, and the module gotcha

- **⚠ #1 GOTCHA: SLURM *and* Pyxis are INVISIBLE unless the module env is loaded.** A bare non-interactive `ssh EmpireAI_Beta 'sinfo'` returns EMPTY (no partitions) and `srun --help` shows 0 container flags. **Always wrap remote commands in a login shell**: `ssh EmpireAI_Beta "bash -lc '<cmd>'"` (sources profile → loads modules → real `sinfo`/`srun`/Pyxis appear). Binaries live at `/cm/local/apps/slurm/current/bin` (Bright Cluster Manager); `SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf`; SLURM **25.05.6**.
- `~/.bashrc` throws `module: command not found` in non-interactive shells (harmless noise — filter it).
- **Account association (SU charging live since 2026-07-01; `AccountingStorageEnforce = associations,limits,qos,safe`).** `bf996` is associated with account **`ny_chinmayh_datacomp`** (DataComp / Chinmay Hegde, NYU), QOS `interactive,long,standard,…`. Submit with `-A ny_chinmayh_datacomp` — jobs run on `beta` (validated: multi-node B200 SFT runs land + train). Live-check: `sacctmgr -n show assoc user=bf996`.
- QoS tiers (from the portal): `test` (0.5× SU, ≤4 GPU/6h, high-prio sandbox), `interactive` (1.0×, 2h), `standard` (1.0×, ≤36 GPU/48h — production default), `long` (0.5×, backfill, ≤7d), `priority` (2.0×, 24h). **SU charging is live** (since 2026-07-01).

## Monitoring + fetching results (ride the operator's socket)

Non-interactive monitoring rides the operator's 2FA ControlMaster socket (see Access). Validated recipe:

- **Every remote command MUST be a login shell** (else SLURM/Pyxis are invisible — the #1 gotcha):
  `ssh -o BatchMode=yes EmpireAI_Beta "bash -lc '<cmd>'"`. Filter the benign header noise
  (`module: command not found`, `Loading gcc/…`, `gmp/ mpfr/ mpc/`).
- **Socket down?** A `Permission denied (publickey…)` or a timeout = the master socket idled out (or was
  never established). I cannot do the 2FA — ask the operator to run `ssh EmpireAI_Beta` in a real terminal
  to re-establish it, then reuse for ~8h.
- **Job liveness:** `squeue -u bf996 -o "%.10i %.24j %.8T %.10M %.6D %R"` (SLURM 25.05; `sacct -u bf996 -S <t>`
  for terminal states). RUNNING is not proof of progress — check the `.out` mtime / latest step.
- **SFT output layout:** `output_dir = /mnt/home/bf996/experiments/<run>/…out/`; per-step metrics live in the
  **latest `checkpoint-<N>/trainer_state.json`** (`log_history[]` = step/loss/grad_norm/lr; `global_step` /
  `max_steps`). Job stdout at `~/logs/<name>_<jobid>.out`. **⚠ Lustre — no `find`/`du` walks; `ls` depth-1.**
- **Fetch results to the Mac:** `scp -o BatchMode=yes EmpireAI_Beta:<output_dir>/checkpoint-<N>/trainer_state.json …`
  into the experiment dir. **Pull only the small JSON metrics** (trainer_state/config/generation_config) —
  NEVER checkpoints / `model.safetensors*` / the 11 MB `tokenizer.json` (not results; too big for the Mac).

## Containerization — Pyxis/Enroot (mandatory)

The compute nodes are a **minimal install**; "treat bare-metal jobs as the exception." All workloads run in containers via **Pyxis + Enroot** (enroot at `/usr/bin/enroot`, ~27 srun `--container-*` flags).
- **Bare `enroot` on the login/mgmt node FAILS** (`mkdir /var/lib/enroot: Permission denied`) — enroot runs on **compute nodes via Pyxis/srun**, not the login node.
- Recipe (from the GB200 NVL72 doc):
  ```bash
  srun --partition=beta --gpus-per-node=4 --segment=<1|4|8|16> \
    --container-image="<image>" \
    --container-mounts="$PROJECT:/workspace,$DATA:/data" \
    --container-remap-root bash -lc "cd /workspace; <cmd>"
  ```
- **Images must be aarch64 + Blackwell-capable** (NGC ARM CUDA/PyTorch ≥ CUDA 12.8). Because the beta login is aarch64, images can be built/imported **natively on-cluster** (no cross-arch qemu — contrast Alpha, whose x86 login can't build the aarch64 compute images). Enroot import a `.sqsh` once to `/ddn` (persistent) and reuse via `--container-image=/ddn/.../img.sqsh`.

## Filesystems (⚠ no `find`/`du` walks — Lustre)

| Mount | Type | Size | Use |
|---|---|---|---|
| `/mnt/home/bf996` (HOME) | Vast NFS | 18PB pool, **100 GB user quota** | code, small configs; **too small for models** |
| **`/ddn`** | **DDN Lustre** (IB o2ib) | **11 PB** | **the data/checkpoint/image FS** — but **NO per-user/group dir provisioned yet** (root-owned; only `/ddn/client_validation` is world-writable) |
| `/tmp` | node-local ext4 | 11 TB | fast per-node scratch (ephemeral) |
| `/cm/shared` | NFS | 3.7 TB | Bright shared apps (read-only-ish) |

> ⚠ **Storage gap:** there is **no `/ddn/<group>/<user>` allocation for us yet** — file a support ticket for a real DDN allocation before any large training. **Interim for build + smoke tests:** enroot image cache in HOME (100GB holds ~1 image) or a self-made dir under `/ddn/client_validation`; scratch/checkpoints on `/tmp` (node-local) or `/ddn/client_validation`. Do NOT rely on `client_validation` for anything persistent (it's a validation scratch, may be wiped).

## Target workflows (containerized)

Only two for now, both via Pyxis/Enroot on `beta`:
1. **SFT with Axolotl** (aarch64 + Blackwell container).
2. **MoE pretraining with Marin** (initialized-MoE pretrain). How the Marin team runs MoE pretraining lives in the **Marin GitHub issues** — search the local Marin mirror via the mumwelt skills (see `.claude/projects/mum/mum.md`; prefer `marin-research`).

## Support / docs

Portal: https://empireai.freshdesk.com/support/home · GB200 NVL72 guide: article `157000373786` · Getting-started: `157000374441`. Admins (Flatiron): Kali McLennan, Geraud Krawezik, Robert Harrison, Ian Fisk.
