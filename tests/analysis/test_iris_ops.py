from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from scripts.iris import coreweave_ops
from scripts.iris.iris_ops import job_bundle, job_id_parts, load_bundle_manifest, write_bundle_manifest
from scripts.iris import watch_coreweave_rl


def test_job_bundle_uses_cluster_and_full_iris_identity(tmp_path):
    bundle = job_bundle(tmp_path, "cw-rno2a", "/benjaminfeuer/glm52-r10")

    assert bundle.directory == tmp_path / "jobs" / "cw-rno2a" / "benjaminfeuer" / "glm52-r10"

    write_bundle_manifest(bundle, {"kind": "harbor", "progress": {"completed": 4}})

    assert json.loads(bundle.manifest_path.read_text())["job_id"] == "/benjaminfeuer/glm52-r10"
    assert load_bundle_manifest(bundle)["progress"] == {"completed": 4}


@pytest.mark.parametrize("job_id", ["glm52-r10", "/benjaminfeuer", "/benjaminfeuer/../bad"])
def test_job_id_parts_rejects_noncanonical_or_unsafe_ids(job_id):
    with pytest.raises(ValueError):
        job_id_parts(job_id)


def test_find_pod_uses_untruncated_iris_job_label(monkeypatch):
    job_id = "/benjaminfeuer/grug-agentic-eval-v2-65k-harbor394c-r3"
    monkeypatch.setattr(
        coreweave_ops,
        "command",
        lambda _args: json.dumps(
            {
                "items": [
                    {
                        "metadata": {
                            "name": "iris-benjaminfeuer-grug-agentic-eval-v2-65k-harbor39-c309bb6c-0",
                            "labels": {"iris.job_id": "benjaminfeuer.grug-agentic-eval-v2-65k-harbor394c-r3"},
                        },
                        "status": {"phase": "Running"},
                    }
                ]
            }
        ),
    )

    assert coreweave_ops.find_pod(["kubectl"], SimpleNamespace(job=job_id, pod=None)) == (
        "iris-benjaminfeuer-grug-agentic-eval-v2-65k-harbor39-c309bb6c-0"
    )


def test_save_ray_logs_reports_empty_tar_stream_as_sync_error(monkeypatch, tmp_path):
    class EmptyTarProcess:
        stdout = BytesIO()
        stderr = BytesIO(b"tar: session log vanished")

        def wait(self) -> int:
            return 1

    monkeypatch.setattr(coreweave_ops.subprocess, "Popen", lambda *_args, **_kwargs: EmptyTarProcess())

    with pytest.raises(RuntimeError, match="Could not archive Ray/vLLM logs"):
        coreweave_ops.save_ray_logs(
            ["kubectl"],
            "pod",
            "task",
            [{"path": "worker-1.out", "size": 10}],
            100,
            tmp_path,
        )
    assert (tmp_path / "ray-vllm-sync-error.txt").read_text() == "tar: session log vanished"


def test_save_ray_logs_retries_transient_kubectl_html(monkeypatch, tmp_path):
    archive_bytes = BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        member = tarfile.TarInfo("worker-1.out")
        payload = b"ray log\n"
        member.size = len(payload)
        archive.addfile(member, BytesIO(payload))

    class TarProcess:
        def __init__(self, stdout: bytes, stderr: bytes, return_code: int):
            self.stdout = BytesIO(stdout)
            self.stderr = BytesIO(stderr)
            self.return_code = return_code

        def wait(self) -> int:
            return self.return_code

    processes = iter(
        [
            TarProcess(b"", b"<!doctype html><html>temporary proxy error", 1),
            TarProcess(archive_bytes.getvalue(), b"", 0),
        ]
    )
    delays: list[int] = []
    monkeypatch.setattr(coreweave_ops.subprocess, "Popen", lambda *_args, **_kwargs: next(processes))
    monkeypatch.setattr(coreweave_ops.time, "sleep", delays.append)

    saved, skipped = coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        [{"path": "worker-1.out", "size": 10}],
        100,
        tmp_path,
    )

    assert saved == [{"path": "worker-1.out", "size": 10}]
    assert skipped == []
    assert delays == [coreweave_ops.DNS_INITIAL_BACKOFF]
    assert (tmp_path / "worker-1.out").read_bytes() == b"ray log\n"


def test_coreweave_command_retries_transient_kubectl_html(monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess([], 1, stderr="<!doctype html><html>temporary proxy error"),
            subprocess.CompletedProcess([], 0, stdout="ok\n"),
        ]
    )
    delays: list[int] = []
    monkeypatch.setattr(coreweave_ops.subprocess, "run", lambda *_args, **_kwargs: next(results))
    monkeypatch.setattr(coreweave_ops.time, "sleep", delays.append)

    assert coreweave_ops.command(["kubectl", "get", "pods"]) == "ok\n"
    assert delays == [coreweave_ops.DNS_INITIAL_BACKOFF]


def test_coreweave_command_retries_generic_kubectl_exec_transport_failure(monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess([], 1, stderr="command terminated with exit code 1"),
            subprocess.CompletedProcess([], 0, stdout="ok\n"),
        ]
    )
    delays: list[int] = []
    monkeypatch.setattr(coreweave_ops.subprocess, "run", lambda *_args, **_kwargs: next(results))
    monkeypatch.setattr(coreweave_ops.time, "sleep", delays.append)

    assert coreweave_ops.command(["kubectl", "exec", "pod", "--", "true"]) == "ok\n"
    assert delays == [coreweave_ops.DNS_INITIAL_BACKOFF]


def test_ray_log_inventory_uses_explicit_python_for_rl_images(monkeypatch):
    monkeypatch.setattr(
        coreweave_ops,
        "resolve_container_python",
        lambda *_args: (_ for _ in ()).throw(AssertionError("generic discovery should not run")),
    )
    commands: list[list[str]] = []

    def fake_command(arguments, **_kwargs):
        commands.append(arguments)
        return "[]"

    monkeypatch.setattr(coreweave_ops, "command", fake_command)

    assert coreweave_ops.ray_log_inventory(
        ["kubectl"], "pod", "task", python_executable="/opt/openthoughts/envs/rl/bin/python"
    ) == []
    assert "/opt/openthoughts/envs/rl/bin/python" in commands[0]


def test_rl_sync_warning_never_renders_proxy_html():
    warning = watch_coreweave_rl.sync_warning(
        ("pod Ray/vLLM: Could not archive logs (<!doctype html><html>proxy body)",)
    )

    assert warning == "Ray/vLLM log sync unavailable; local diagnostic saved"


def test_rl_progress_reporter_writes_phase_and_elapsed_time(monkeypatch, capsys):
    monkeypatch.setattr(watch_coreweave_rl.time, "monotonic", lambda: 165.0)

    reporter = watch_coreweave_rl.ProgressReporter(started_at=100.0)
    reporter.phase("trace inventory 1/2")

    assert capsys.readouterr().err == "[rl-watch +01:05] trace inventory 1/2\n"


def test_rl_discovery_skips_iris_preamble_and_includes_recent_terminal_jobs(monkeypatch):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    output = """I20260722 controller tunnel ready
job_id,state,submitted_at_ms,finished_at_ms,entrypoint_json
/benjaminfeuer/rl-live,3,1000,,"start_rl_iris_controller.py --train_data '[\\"live\\"]'"
/benjaminfeuer/rl-failed,5,900,1500,"start_rl_iris_controller.py --train_data '[\\"failed\\"]'"
"""
    queries: list[str] = []

    def fake_run_iris(_cluster, arguments, **_kwargs):
        queries.append(arguments[1])
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    monkeypatch.setattr(watch_coreweave_rl, "run_iris", fake_run_iris)

    jobs, errors = watch_coreweave_rl.discover_rl_jobs(
        cluster,
        "benjaminfeuer",
        terminal_since_ms=1200,
    )

    assert errors == []
    assert [job.short_name for job in jobs] == ["rl-live", "rl-failed"]
    assert [job.is_terminal for job in jobs] == [False, True]
    assert "j.state IN (4,5) AND j.finished_at_ms >= 1200" in queries[0]


def test_recent_trace_jobs_uses_remote_last_modified_and_preserves_remote_counts():
    root = "iris/rl/trace_jobs/"
    first = datetime(2026, 7, 22, 10, tzinfo=UTC)
    objects = [
        {"Key": f"{root}z-old/result.json", "Size": 1, "LastModified": first},
        {"Key": f"{root}a-new/agent/trajectory.json", "Size": 1, "LastModified": first + timedelta(hours=3)},
        {"Key": f"{root}a-new/result.json", "Size": 1, "LastModified": first + timedelta(hours=2)},
        {"Key": f"{root}m-middle/result.json", "Size": 1, "LastModified": first + timedelta(hours=2)},
    ]

    selected, available, completed = watch_coreweave_rl.recent_trace_jobs(objects, root, trace_sync_limit=2)

    assert [trace.name for trace in selected] == ["a-new", "m-middle"]
    assert available == 3
    assert completed == 3
    full, _, _ = watch_coreweave_rl.recent_trace_jobs(objects, root, trace_sync_limit=0)
    assert [trace.name for trace in full] == ["a-new", "m-middle", "z-old"]

    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    first_job = watch_coreweave_rl.RlJob(cluster, "/user/first", "running", 0, "")
    second_job = watch_coreweave_rl.RlJob(cluster, "/user/second", "running", 0, "")
    first_inventory = watch_coreweave_rl.TraceInventory(first_job, "bucket", root, object(), tuple(full), 3, 3)
    second_inventory = watch_coreweave_rl.TraceInventory(
        second_job,
        "bucket",
        root,
        object(),
        (watch_coreweave_rl.TraceJobObjects("other", first + timedelta(hours=4), (), True),),
        1,
        1,
    )

    fleet_selected = watch_coreweave_rl.select_recent_fleet_traces(
        [first_inventory, second_inventory], trace_sync_limit=2
    )

    assert [trace.name for trace in fleet_selected[("cw-rno2a", "/user/first")]] == ["a-new"]
    assert [trace.name for trace in fleet_selected[("cw-rno2a", "/user/second")]] == ["other"]
    manifest = watch_coreweave_rl.trace_selection_manifest(
        fleet_selected[("cw-rno2a", "/user/first")], first_inventory, 2, fleet_available=4, fleet_selected=2
    )
    assert manifest["selection"] == "latest_object_store_last_modified_across_active_rl_jobs"
    assert manifest["omitted_traces"] == 2
    assert manifest["fleet_selected_traces"] == 2

    full_fleet = watch_coreweave_rl.select_recent_fleet_traces(
        [first_inventory, second_inventory], trace_sync_limit=0
    )
    assert sum(len(traces) for traces in full_fleet.values()) == 4


def test_recent_trace_jobs_requires_remote_last_modified_metadata():
    with pytest.raises(ValueError, match="LastModified"):
        watch_coreweave_rl.recent_trace_jobs(
            [{"Key": "iris/rl/trace_jobs/trace/result.json", "Size": 1}],
            "iris/rl/trace_jobs/",
            trace_sync_limit=500,
        )
