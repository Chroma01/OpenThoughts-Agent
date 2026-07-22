#!/usr/bin/env python3
"""Sync and summarize every active CoreWeave Iris RL job for one user.

The monitor is deliberately read-only.  Each Iris job gets one stable local
directory, keyed by its full Iris job id, so repeated sweeps and Iris task
retries refresh the same artifacts instead of making timestamped copies.  It
captures the complete finelog plus complete pod/Ray/vLLM logs, then mirrors
the 500 most recently modified Harbor ``trace_jobs`` by default. The recent
trace selection is based on object-store ``LastModified`` metadata, never trace
names. Use ``--trace-sync-limit 0`` for a deliberately full trace sync. The
sync still skips non-log objects larger than the configured size bound; that
rule avoids repeatedly downloading giant rollout payloads while preserving any
diagnostic log regardless of size.

By default the scope is the current lab user's active RL jobs on both
CoreWeave GPU clusters.  Use ``--all-users`` only when cross-user monitoring is
intended.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.iris.coreweave_ops import (  # noqa: E402
    CLUSTERS as COREWEAVE_CLUSTERS,
    NAMESPACE,
    kubectl_base,
    object_store_client,
    ray_log_inventory,
    safe_relative_path,
    save_ray_logs,
    split_s3_uri,
)
from scripts.iris.iris_ops import (  # noqa: E402
    DEFAULT_BUNDLE_ROOT,
    job_bundle,
    run_iris_command,
    write_bundle_manifest,
)


DEFAULT_USER = "benjaminfeuer"
DEFAULT_MAX_NON_LOG_BYTES = 100 * 1024 * 1024
DEFAULT_TRACE_SYNC_LIMIT = 500
ACTIVE_STATES = {1: "pending", 2: "building", 3: "running"}
STATE_NAMES = {
    **ACTIVE_STATES,
    4: "succeeded",
    5: "failed",
    6: "killed",
    7: "worker_failed",
    8: "unschedulable",
}
RL_ENTRYPOINT_MARKERS = ("start_rl_iris_controller.py", "cloud.iris.run_rl")
TRIALS_URI_PATTERN = re.compile(
    r"(?:terminal_bench_config\.trials_dir=|--trials-dir(?:=|\s+))"
    r"(?P<uri>s3://[^\s'\"\\]+)"
)
TRAIN_DATA_PATTERN = re.compile(
    r"--train[_-]data(?:=|\s+)(?:'(?P<single>\[[^']+\])'|\"(?P<double>\[[^\"]+\])\"|(?P<bare>\[[^\s]+\]))"
)
PROGRESS_PATTERN = re.compile(r"Training Step Progress:\s*(\d+)\s*/\s*(\d+)")
MIRROR_PATTERN = re.compile(r"WANDB_MIRROR kind=train step=(\d+) metrics=(\{.*\})")
ERROR_PATTERNS = (
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"(?:RayTaskError|ActorDiedError|WorkerCrashedError)"),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"Train loop failed", re.IGNORECASE),
)
LOG_SUFFIXES = (".log", ".out", ".err", ".jsonl", ".txt")


@dataclass(frozen=True)
class Cluster:
    name: str
    kubeconfig: Path
    context: str | None


@dataclass(frozen=True)
class RlJob:
    cluster: Cluster
    job_id: str
    state: str
    submitted_at_ms: int
    entrypoint: str
    dataset: str = "—"

    @property
    def short_name(self) -> str:
        return self.job_id.rstrip("/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ArtifactResult:
    finelog: str
    pod_logs: str
    ray_logs: str
    traces: str
    trace_started: int | None
    trace_completed: int | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class TraceJobObjects:
    """One remote trace directory and its latest object modification time."""

    name: str
    last_modified: datetime
    objects: tuple[dict[str, Any], ...]
    completed: bool


CLUSTERS = tuple(
    Cluster(name, config.kubeconfig, config.context)
    for name, config in COREWEAVE_CLUSTERS.items()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Root for canonical local Iris evidence bundles and RL reports.",
    )
    parser.add_argument("--user", default=DEFAULT_USER, help=f"Iris user to monitor (default: {DEFAULT_USER}).")
    parser.add_argument("--all-users", action="store_true", help="Discover active RL jobs for every user.")
    parser.add_argument(
        "--max-non-log-bytes",
        type=int,
        default=DEFAULT_MAX_NON_LOG_BYTES,
        help="Skip non-log trace objects larger than this many bytes (default: 100 MiB; 0 disables).",
    )
    parser.add_argument(
        "--trace-sync-limit",
        type=int,
        default=DEFAULT_TRACE_SYNC_LIMIT,
        help=(
            "Sync only this many most recently modified trace directories per RL job "
            "(default: 500; 0 syncs every remote trace)."
        ),
    )
    parser.add_argument("--no-sync", action="store_true", help="Report lifecycle state without collecting artifacts.")
    return parser.parse_args()


def run_iris(cluster: Cluster, arguments: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["KUBECONFIG"] = str(cluster.kubeconfig)
    return run_iris_command(
        arguments,
        cluster=cluster.name,
        iris_bin="/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris",
        environment=environment,
        timeout=timeout,
    )


def entrypoint_text(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw))
    except json.JSONDecodeError:
        return raw


def command_strings(entrypoint: str) -> list[str]:
    """Return all string leaves from an Iris entrypoint JSON payload."""
    try:
        value = json.loads(entrypoint)
    except json.JSONDecodeError:
        return [entrypoint]

    strings: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return strings


def dataset_from_entrypoint(entrypoint: str) -> str:
    """Extract and deduplicate the submitted dataset list without guessing from a config."""
    datasets: list[str] = []
    for command in command_strings(entrypoint):
        for match in TRAIN_DATA_PATTERN.finditer(command):
            raw_dataset_list = next(value for value in match.groupdict().values() if value is not None)
            try:
                values = json.loads(raw_dataset_list)
            except json.JSONDecodeError:
                continue
            if isinstance(values, list):
                datasets.extend(str(value) for value in values)
    return ", ".join(dict.fromkeys(datasets)) or "—"


def discover_rl_jobs(cluster: Cluster, user: str | None) -> tuple[list[RlJob], list[str]]:
    where_user = "" if user is None else f" AND j.job_id LIKE '/{user}/%'"
    sql = (
        "SELECT j.job_id, j.state, j.submitted_at_ms, jc.entrypoint_json "
        "FROM jobs j JOIN job_config jc ON j.job_id=jc.job_id "
        f"WHERE j.state IN ({','.join(str(state) for state in sorted(ACTIVE_STATES))}){where_user} "
        "ORDER BY j.submitted_at_ms DESC"
    )
    result = run_iris(cluster, ["query", sql, "-f", "csv"])
    if result.returncode:
        message = (result.stderr or result.stdout).strip().replace("\n", " ")
        return [], [f"{cluster.name}: discovery failed: {message[-240:]}"]

    jobs: list[RlJob] = []
    for row in csv.DictReader(result.stdout.splitlines()):
        entrypoint = entrypoint_text(row.get("entrypoint_json", ""))
        if not any(marker in entrypoint for marker in RL_ENTRYPOINT_MARKERS):
            continue
        try:
            state_code = int(row["state"])
            submitted_at_ms = int(row["submitted_at_ms"])
        except (KeyError, ValueError):
            continue
        jobs.append(
            RlJob(
                cluster=cluster,
                job_id=row["job_id"],
                state=STATE_NAMES.get(state_code, f"state-{state_code}"),
                submitted_at_ms=submitted_at_ms,
                entrypoint=entrypoint,
                dataset=dataset_from_entrypoint(entrypoint),
            )
        )
    return jobs, []


def job_directory(bundle_root: Path, job: RlJob) -> Path:
    """Return the shared canonical evidence directory for this Iris job."""
    return job_bundle(bundle_root, job.cluster.name, job.job_id).directory


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def fetch_finelog(job: RlJob, destination: Path) -> tuple[str, str | None]:
    result = run_iris(
        job.cluster,
        ["job", "logs", job.job_id, "--max-lines", "10000000", "--no-tail"],
        timeout=900,
    )
    stderr_path = destination / "finelog.stderr"
    stderr_path.write_text(result.stderr)
    if result.returncode:
        message = (result.stderr or result.stdout).strip().replace("\n", " ")
        return "unavailable", f"finelog: {message[-180:]}"
    (destination / "finelog.log").write_text(result.stdout)
    return f"{len(result.stdout.splitlines()):,} lines", None


def job_pods(job: RlJob) -> list[tuple[str, str]]:
    base = kubectl_base(
        COREWEAVE_CLUSTERS[job.cluster.name], SimpleNamespace(kubeconfig=None, kube_context=None)
    )
    result = subprocess.run(
        [*base, "-n", NAMESPACE, "get", "pods", "-o", "json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-240:])
    needle = job.short_name.lower()
    return sorted(
        (
            item["metadata"]["name"],
            item.get("status", {}).get("phase", "Unknown"),
        )
        for item in json.loads(result.stdout).get("items", [])
        if needle in item.get("metadata", {}).get("name", "").lower()
    )


def fetch_complete_pod_log(base: list[str], pod: str, destination: Path) -> None:
    result = subprocess.run(
        [*base, "-n", NAMESPACE, "logs", pod, "-c", "task", "--tail=-1"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    destination.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-240:])


def fetch_complete_ray_logs(base: list[str], pod: str, destination: Path) -> int:
    inventory = ray_log_inventory(base, pod, "task", patterns=None)
    if not inventory:
        return 0
    try:
        saved, skipped = save_ray_logs(base, pod, "task", inventory, sys.maxsize, destination)
    except RuntimeError:
        # Ray rotates/removes worker logs while a live pod is writing. Rebuild
        # the inventory once so a stale path cannot abort the whole fleet scan.
        inventory = ray_log_inventory(base, pod, "task", patterns=None)
        if not inventory:
            return 0
        saved, skipped = save_ray_logs(base, pod, "task", inventory, sys.maxsize, destination)
    if skipped:
        raise AssertionError("A maximum-size sync should not skip Ray/vLLM logs.")
    return len(saved)


def sync_pod_and_ray_logs(job: RlJob, destination: Path) -> tuple[str, str, list[str]]:
    """Capture all current pod stdout plus all Ray/vLLM logs without a size cap."""
    errors: list[str] = []
    try:
        pods = job_pods(job)
    except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return "unavailable", "unavailable", [f"pod discovery: {error}"]
    if not pods:
        return "no pod yet", "no pod yet", []
    running_pods = [pod for pod, phase in pods if phase == "Running"]
    if not running_pods:
        phases = ", ".join(sorted(phase for _, phase in pods))
        return f"{len(pods)} pod(s): {phases}", "awaiting host", []

    base = kubectl_base(
        COREWEAVE_CLUSTERS[job.cluster.name], SimpleNamespace(kubeconfig=None, kube_context=None)
    )
    pod_dir = destination / "pod_logs"
    ray_dir = destination / "ray_vllm_logs"
    pod_dir.mkdir(exist_ok=True)
    ray_dir.mkdir(exist_ok=True)
    ray_files = 0
    for pod in running_pods:
        try:
            fetch_complete_pod_log(base, pod, pod_dir / f"{pod}.log")
        except (RuntimeError, subprocess.SubprocessError) as error:
            errors.append(f"{pod} stdout: {error}")
        try:
            pod_ray_dir = ray_dir / pod
            pod_ray_dir.mkdir(exist_ok=True)
            ray_files += fetch_complete_ray_logs(base, pod, pod_ray_dir)
        except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            errors.append(f"{pod} Ray/vLLM: {error}")
    return f"{len(pods)} pod(s), {len(running_pods)} Running", f"{ray_files:,} files", errors


def trials_uri(job: RlJob) -> str:
    match = TRIALS_URI_PATTERN.search(job.entrypoint)
    if match:
        return match.group("uri").rstrip("/" )
    # This is exactly the launcher's --trials-dir auto convention.  Keep the
    # fallback local and visible rather than reading a possibly different YAML.
    return f"s3://marin-us-east-02a/iris/{job.short_name}/trace_jobs"


def is_log_object(relative_path: str) -> bool:
    path = relative_path.lower()
    filename = path.rsplit("/", 1)[-1]
    return (
        any(suffix in filename for suffix in LOG_SUFFIXES)
        or "/logs/" in path
        or path.startswith("logs/")
    )


def recent_trace_jobs(
    objects: list[dict[str, Any]], root_prefix: str, trace_sync_limit: int
) -> tuple[list[TraceJobObjects], int, int]:
    """Return trace directories ordered by latest remote object modification.

    Object-store ``LastModified`` is the only ordering source. A trace is
    complete when any object in its directory is ``result.json``; those counts
    cover the complete remote prefix even when a recent subset is downloaded.
    """
    traces: dict[str, list[dict[str, Any]]] = {}
    for item in objects:
        relative = item["Key"].removeprefix(root_prefix)
        if relative:
            traces.setdefault(relative.split("/", 1)[0], []).append(item)

    trace_jobs: list[TraceJobObjects] = []
    completed = 0
    for name, trace_objects in traces.items():
        latest_modified: datetime | None = None
        completed_trace = False
        for item in trace_objects:
            modified = item.get("LastModified")
            if not isinstance(modified, datetime):
                raise ValueError(f"Object {item['Key']!r} is missing LastModified metadata")
            if latest_modified is None or modified > latest_modified:
                latest_modified = modified
            relative = item["Key"].removeprefix(root_prefix)
            completed_trace = completed_trace or relative.endswith("/result.json")
        assert latest_modified is not None
        completed += completed_trace
        trace_jobs.append(TraceJobObjects(name, latest_modified, tuple(trace_objects), completed_trace))

    trace_jobs.sort(key=lambda trace: (trace.last_modified, trace.name), reverse=True)
    selected = trace_jobs if trace_sync_limit == 0 else trace_jobs[:trace_sync_limit]
    return selected, len(trace_jobs), completed


def trace_selection_manifest(
    selected: list[TraceJobObjects], available: int, trace_sync_limit: int
) -> dict[str, Any]:
    """Describe the bounded selection without recording every omitted trace."""
    return {
        "selection": "latest_object_store_last_modified",
        "trace_sync_limit": trace_sync_limit,
        "available_traces": available,
        "selected_traces": len(selected),
        "omitted_traces": available - len(selected),
        "selected": [
            {"name": trace.name, "last_modified": trace.last_modified.isoformat(), "completed": trace.completed}
            for trace in selected
        ],
    }


def sync_trace_jobs(
    job: RlJob, destination: Path, max_non_log_bytes: int, trace_sync_limit: int
) -> tuple[str, int, int, str | None]:
    uri = trials_uri(job)
    bucket, prefix = split_s3_uri(uri)
    base = kubectl_base(
        COREWEAVE_CLUSTERS[job.cluster.name], SimpleNamespace(kubeconfig=None, kube_context=None)
    )
    client = object_store_client(base, COREWEAVE_CLUSTERS[job.cluster.name])
    root_prefix = f"{prefix.rstrip('/')}/"
    destination.mkdir(exist_ok=True)
    copied = skipped = available = completed = selected_count = 0
    skipped_objects: list[dict[str, Any]] = []
    try:
        paginator = client.get_paginator("list_objects_v2")
        objects: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=root_prefix):
            objects.extend(page.get("Contents", []))
        selected, available, completed = recent_trace_jobs(objects, root_prefix, trace_sync_limit)
        selected_count = len(selected)
        write_json(destination / "sync_selection.json", trace_selection_manifest(selected, available, trace_sync_limit))
        for trace in selected:
            for item in trace.objects:
                relative = item["Key"].removeprefix(root_prefix)
                size = int(item["Size"])
                if max_non_log_bytes and size > max_non_log_bytes and not is_log_object(relative):
                    skipped += 1
                    skipped_objects.append({"key": relative, "size": size, "reason": "non_log_size_limit"})
                    continue
                local_path = destination / safe_relative_path(item["Key"], root_prefix)
                if local_path.exists() and local_path.stat().st_size == size:
                    continue
                local_path.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, item["Key"], str(local_path))
                copied += 1
    except Exception as error:  # object stores may race a currently-uploading trial
        write_json(destination / "skipped_objects.json", skipped_objects)
        scope = f"newest {selected_count:,}/{available:,} traces; " if available else ""
        return f"partial: {scope}{copied:,} copied, {skipped:,} skipped", available, completed, str(error)[-240:]
    write_json(destination / "skipped_objects.json", skipped_objects)
    scope = f"all {available:,}" if trace_sync_limit == 0 else f"newest {len(selected):,}/{available:,}"
    return f"{scope} traces; {copied:,} copied, {skipped:,} skipped", available, completed, None


def parse_metrics(finelog: Path) -> tuple[int | None, int | None, dict[str, Any], str | None]:
    if not finelog.exists():
        return None, None, {}, None
    try:
        text = finelog.read_text(errors="replace")
    except OSError as error:
        return None, None, {}, f"could not read finelog: {error}"
    progress = PROGRESS_PATTERN.findall(text)
    step = int(progress[-1][0]) if progress else None
    total = int(progress[-1][1]) if progress else None
    for line in reversed(text.splitlines()):
        match = MIRROR_PATTERN.search(line)
        if not match:
            continue
        try:
            metrics = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        return int(match.group(1)), total, metrics, None
    return step, total, {}, None


def metric(metrics: dict[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def display_metric(value: Any | None, precision: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def terminal_signal(finelog: Path) -> str | None:
    if not finelog.exists():
        return None
    try:
        tail = finelog.read_text(errors="replace")[-2_000_000:]
    except OSError:
        return None
    for pattern in ERROR_PATTERNS:
        match = pattern.search(tail)
        if match:
            return match.group(0)
    return None


def sync_warning(errors: tuple[str, ...]) -> str | None:
    """Render a stable table cell for artifact-sync errors, never raw proxy bodies."""
    if not errors:
        return None
    first_error = errors[0]
    if "Ray/vLLM" in first_error:
        return "Ray/vLLM log sync unavailable; local diagnostic saved"
    return first_error[-90:]


def box_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def line(values: list[str]) -> str:
        return "│" + "│".join(f" {value.ljust(width)} " for value, width in zip(values, widths)) + "│"

    return "\n".join([border("┌", "┬", "┐"), line(headers), border("├", "┼", "┤"), *(line(row) for row in rows), border("└", "┴", "┘")])


def report_row(job: RlJob, artifacts: ArtifactResult, directory: Path) -> list[str]:
    step, total, metrics, parse_error = parse_metrics(directory / "finelog.log")
    step_display = "—" if step is None else f"{step}/{total if total is not None else '—'}"
    reward = metric(metrics, "reward/avg_raw_reward", "loss/avg_final_rewards")
    policy_loss = metric(metrics, "policy/policy_loss", "policy_loss")
    grad_norm = metric(metrics, "policy/raw_grad_norm", "raw_grad_norm")
    entropy = metric(metrics, "policy/policy_entropy", "policy_entropy")
    log_ratio = metric(metrics, "tis/log_ratio_abs_mean", "log_ratio_abs_mean")
    signal = terminal_signal(directory / "finelog.log")
    trend = f"{job.state.upper()}; entropy={display_metric(entropy)}; TIS log-ratio={display_metric(log_ratio)}"
    if signal:
        trend = f"{job.state.upper()} signal: {signal}; entropy={display_metric(entropy)}; TIS log-ratio={display_metric(log_ratio)}"
    elif step is None:
        trend += "; bring-up/buffer (metrics not emitted)"
    warning = sync_warning(artifacts.errors)
    if warning:
        trend += f"; sync warning: {warning}"
    if parse_error:
        trend += f"; parse warning: {parse_error}"
    return [
        f"{job.cluster.name}/{job.short_name}",
        job.dataset,
        step_display,
        display_metric(reward),
        display_metric(policy_loss),
        display_metric(grad_norm),
        artifacts.traces,
        trend,
    ]


def sync_job(
    job: RlJob, bundle_root: Path, max_non_log_bytes: int, trace_sync_limit: int, *, no_sync: bool
) -> tuple[ArtifactResult, Path]:
    bundle = job_bundle(bundle_root, job.cluster.name, job.job_id)
    directory = bundle.directory
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "job": asdict(job),
        "job_directory": str(directory),
        "trials_uri": trials_uri(job),
        "synced_at": datetime.now(UTC).isoformat(),
        "max_non_log_bytes": max_non_log_bytes,
        "trace_sync_limit": trace_sync_limit,
    }
    if no_sync:
        artifacts = ArtifactResult("not requested", "not requested", "not requested", "not requested", None, None, ())
    else:
        errors: list[str] = []
        finelog, error = fetch_finelog(job, directory)
        if error:
            errors.append(error)
        pod_logs, ray_logs, pod_errors = sync_pod_and_ray_logs(job, directory)
        errors.extend(pod_errors)
        traces, started, completed, trace_error = sync_trace_jobs(
            job, directory / "trace_jobs", max_non_log_bytes, trace_sync_limit
        )
        if trace_error:
            errors.append(f"trace sync: {trace_error}")
        artifacts = ArtifactResult(finelog, pod_logs, ray_logs, traces, started, completed, tuple(errors))
    manifest["artifacts"] = asdict(artifacts)
    write_bundle_manifest(bundle, {"kind": "rl", **manifest})
    return artifacts, directory


def main() -> int:
    args = parse_args()
    if args.max_non_log_bytes < 0:
        raise ValueError("--max-non-log-bytes must be non-negative")
    if args.trace_sync_limit < 0:
        raise ValueError("--trace-sync-limit must be non-negative")
    if not args.all_users and not re.fullmatch(r"[A-Za-z0-9_-]+", args.user):
        raise ValueError("--user may contain only letters, numbers, _ and -")
    args.bundle_root.mkdir(parents=True, exist_ok=True)
    report_directory = args.bundle_root / "reports" / "rl"
    report_directory.mkdir(parents=True, exist_ok=True)

    jobs: list[RlJob] = []
    errors: list[str] = []
    scope_user = None if args.all_users else args.user
    for cluster in CLUSTERS:
        found, discovery_errors = discover_rl_jobs(cluster, scope_user)
        jobs.extend(found)
        errors.extend(discovery_errors)

    rows: list[list[str]] = []
    job_report: dict[str, Any] = {}
    for job in sorted(jobs, key=lambda item: (item.cluster.name, item.submitted_at_ms, item.job_id)):
        try:
            artifacts, directory = sync_job(
                job,
                args.bundle_root,
                args.max_non_log_bytes,
                args.trace_sync_limit,
                no_sync=args.no_sync,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
            directory = job_directory(args.bundle_root, job)
            artifacts = ArtifactResult("unavailable", "unavailable", "unavailable", "unavailable", None, None, (str(error),))
        rows.append(report_row(job, artifacts, directory))
        job_report[job.job_id] = {"cluster": job.cluster.name, "directory": str(directory), "artifacts": asdict(artifacts)}

    table = (
        box_table(["Job", "Dataset", "Step", "Reward", "Policy Loss", "Grad Norm", "Traces", "Trend"], rows)
        if rows
        else "No active CoreWeave Iris RL jobs discovered."
    )
    checked_at = datetime.now(UTC)
    report = f"# Iris CoreWeave RL status — {checked_at.isoformat()}\n\n{table}\n"
    if errors:
        report += "\n## Monitor errors\n\n" + "\n".join(f"- {error}" for error in errors) + "\n"
    timestamp = checked_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = report_directory / f"{timestamp}.md"
    report_path.write_text(report)
    (report_directory / "latest.md").write_text(report)
    write_json(report_directory / "latest.json", {"checked_at": checked_at.isoformat(), "jobs": job_report, "report": str(report_path)})
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
