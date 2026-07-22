#!/usr/bin/env python3
"""Canonical Iris-worker launcher for model-mirror routes."""

from __future__ import annotations

import sys
from typing import Sequence

from hpc.launch_utils import PROJECT_ROOT
from scripts.iris.launch_gcs_to_s3 import GcsToS3Launcher
from scripts.iris.launch_hf_mirror import HfMirrorIrisLauncher

LAUNCHERS = {
    "hf-to-gcs": (HfMirrorIrisLauncher, "Mirror HF model repos to GCS via an Iris worker."),
    "gcs-to-s3": (GcsToS3Launcher, "Mirror staged GCS model repos to S3 via an Iris worker."),
}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(__doc__)
        print(f"Routes: {', '.join(sorted(LAUNCHERS))}")
        return 0
    route, *remaining = arguments
    if route not in LAUNCHERS:
        raise SystemExit(f"Unknown mirror launch route {route!r}; choose one of: {', '.join(sorted(LAUNCHERS))}")
    launcher_type, description = LAUNCHERS[route]
    launcher = launcher_type(PROJECT_ROOT)
    parser = launcher.create_argument_parser(description=description)
    return launcher.run(parser.parse_args(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
