"""The fleet's queue that builds a release's CUDA and PJRT plugins (dew/build_plugin.sh), one job a machine, which
`node submit` takes from beside this script (its scripts are staged from here).

    python3 dew/plugin_jobs.py VERSION SHA > dew/plugin-jobs.json && node submit dew/plugin-jobs.json

A plugin build holds no GPU: Bazel's CUDA toolchain is hermetic and Dew's remote cache serves its objects, so a build
is minutes of CPU (Main and the owner, 2026-11-08 and 2026-11-10: post67's to post73's builds held an exclusive A100
and the GH200 only because a job file was copied). Each machine's job shares a node of its machine, an A100's for
x86_64 and the held GH200 for aarch64 (never `exclusive`, no `gpus`), its CPUs capped (`taskset`) and declared
(`cpus`) so other work keeps the rest, the cache's token staged from bazel.token beside this script (gitignored): an
armada task's disk holds no Bazel output base beside uv's cache (2026-11-10: post74's x86_64 build ran out of space
fetching its external repositories)."""

from __future__ import annotations

import json
import re
import sys

FORK = "https://github.com/AshishKumar4/jax"
CORES = 24
"""The GH200's cores an aarch64 build takes, of its 72: Bazel's local actions are few with the remote cache warm."""


def jobs(version: str, sha: str) -> dict:
    """The queue that builds `version`'s plugins from the fork's commit `sha`, for x86_64 and aarch64."""
    if not re.fullmatch(r"\d+(\.\d+)+(\.post\d+)?", version) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit(f"plugin_jobs: {version!r} is no release version or {sha!r} no full commit sha")
    tag = version.rsplit(".", 1)[-1].upper()
    common = {"ref": "main", "keep": [], "priority": 3, "prepare": f"git ls-remote {FORK} {sha}",
              "measures": f"{version}'s CUDA and PJRT plugins, every Python and CUDA, for its machine"}
    build = f"bash $SESSION/extra/build_plugin.sh {sha} {version} \"$OUT\""
    return {"note": f"{version}'s plugin builds (dew/plugin_jobs.py): no GPU held", "jobs": [
        {"name": f"PLUGIN-{tag}-{machine}", "kind": "correctness", "gpu": gpu, "cpus": CORES, "minutes": 120,
         **common, "run": f"DEW_BAZEL_TOKEN_FILE=$SESSION/extra/bazel.token taskset -c 0-{CORES - 1} {build}"}
        for machine, gpu in (("x86_64", "a100"), ("aarch64", "gh200"))
    ]}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: plugin_jobs.py VERSION SHA")
    json.dump(jobs(sys.argv[1], sys.argv[2]), sys.stdout, indent=1)
    print()
