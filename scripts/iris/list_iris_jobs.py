#!/usr/bin/env python3
"""List one Iris user's recent jobs in chronological order (oldest first)."""

from __future__ import annotations

import argparse
import csv
import getpass
import io
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from scripts.iris.iris_ops import DEFAULT_CLUSTER, STATE_NAMES, run_iris_command

USER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
STATE_LABELS = {"killed": "terminated", "worker_failed": "worker failed"}
COREWEAVE_KUBECONFIGS = {
    "cw-rno2a": Path("/Users/benjaminfeuer/.kube/coreweave-iris"),
    "cw-us-east-02a": Path("/Users/benjaminfeuer/.kube/coreweave-iris-gpu"),
}


def command_environment(cluster: str) -> dict[str, str] | None:
    """Return the local kubeconfig override required for a CoreWeave cluster."""
    kubeconfig = COREWEAVE_KUBECONFIGS.get(cluster)
    if kubeconfig is None:
        return None
    environment = os.environ.copy()
    environment["KUBECONFIG"] = str(kubeconfig)
    return environment


def classify_job_type(job_id: str) -> str:
    """Return a conservative workload hint based only on the submitted job name."""
    name = job_id.lower()
    if any(hint in name for hint in ("mirror", "gcs2s3", "hf2s3")):
        return "mirror"
    if any(hint in name for hint in ("skyrl", "grpo", "rl-", "-rl", "terminus")):
        return "RL"
    if any(hint in name for hint in ("eval", "tb2", "terminal-bench", "swebench")):
        return "eval"
    if any(hint in name for hint in ("tracegen", "datagen", "harbor", "pilot")):
        return "datagen"
    if any(hint in name for hint in ("sft", "finetune", "finetuning")):
        return "SFT"
    if any(hint in name for hint in ("serve", "vllm", "endpoint")):
        return "serve"
    return "other"


def state_label(value: str | int | None) -> str:
    """Normalize a numeric controller state into a readable lifecycle state."""
    try:
        state = STATE_NAMES.get(int(value), f"state {value}")
    except (TypeError, ValueError):
        state = str(value or "unknown").lower()
    return STATE_LABELS.get(state, state.replace("_", " "))


def parse_jobs_csv(output: str) -> list[dict[str, str]]:
    """Parse Iris CSV while discarding the CLI's informational preamble."""
    lines = output.splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("job_id,")), None)
    if header_index is None:
        raise ValueError("Iris query returned no CSV job_id header")
    return list(csv.DictReader(io.StringIO("\n".join(lines[header_index:]))))


def query_jobs(*, user: str, hours: float, cluster: str, now_ms: int | None = None) -> list[dict[str, str]]:
    """Return one user's submitted jobs since the bounded epoch cutoff."""
    if not USER_RE.fullmatch(user):
        raise ValueError(f"Invalid Iris user {user!r}")
    if hours <= 0:
        raise ValueError("--hours must be positive")
    cutoff_ms = (now_ms if now_ms is not None else int(time.time() * 1000)) - int(hours * 3_600_000)
    sql = (
        "SELECT job_id,state,submitted_at_ms,started_at_ms,finished_at_ms,error,exit_code "
        "FROM jobs "
        f"WHERE job_id LIKE '/{user}/%' AND submitted_at_ms >= {cutoff_ms} "
        "ORDER BY submitted_at_ms ASC"
    )
    result = run_iris_command(
        ["query", sql, "-f", "csv"],
        cluster=cluster,
        environment=command_environment(cluster),
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Iris jobs query failed: {detail[-600:]}")
    return parse_jobs_csv(result.stdout)


def _timestamp(ms: str | None) -> str:
    try:
        return datetime.fromtimestamp(int(ms or "") / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "—"


def _short(value: str, width: int) -> str:
    return value if len(value) <= width else f"{value[:width - 1]}…"


def render_table(rows: list[dict[str, str]]) -> str:
    """Render controller rows as a stable oldest-first Unicode table."""
    headers = ("Submitted UTC", "Job", "Type (hint)", "State", "Exit", "Error")
    values = [
        (
            _timestamp(row.get("submitted_at_ms")),
            _short(row.get("job_id", ""), 54),
            classify_job_type(row.get("job_id", "")),
            state_label(row.get("state")),
            row.get("exit_code") or "—",
            _short((row.get("error") or "").replace("\n", " "), 54) or "—",
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for value in values:
        widths = [max(width, len(cell)) for width, cell in zip(widths, value)]
    def border(left: str, join: str, right: str) -> str:
        return left + join.join("─" * (width + 2) for width in widths) + right
    lines = [border("┌", "┬", "┐"), "│ " + " │ ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " │", border("├", "┼", "┤")]
    lines.extend("│ " + " │ ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " │" for row in values)
    lines.append(border("└", "┴", "┘"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=getpass.getuser(), help="Iris username (default: current OS user).")
    parser.add_argument("--hours", type=float, default=24.0, help="Submitted-job window in hours (default: 24).")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER, help=f"Iris cluster (default: {DEFAULT_CLUSTER}).")
    args = parser.parse_args(argv)
    try:
        rows = query_jobs(user=args.user, hours=args.hours, cluster=args.cluster)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"# Iris jobs — user={args.user} cluster={args.cluster} last={args.hours:g}h; oldest first")
    print(render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
