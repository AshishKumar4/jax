"""The fleet's queue that builds a release's CUDA and PJRT plugins (dew/build_plugin.sh), one job a machine, which
`node submit` takes from beside this script (its scripts are staged from here).

    python3 dew/plugin_jobs.py VERSION SHA > dew/plugin-jobs.json && node submit dew/plugin-jobs.json

A plugin build holds no GPU: Bazel's CUDA toolchain is hermetic and Dew's remote cache serves its objects, so a build
is minutes of CPU (Main and the owner, 2026-11-08 and 2026-11-10: post67's to post73's builds held an exclusive A100
and the GH200 only because a job file was copied). The x86_64 wheels build on armada's CPUs (`kind: build`), the
cache's token the armada secret DEW_BAZEL_TOKEN. The aarch64 wheels need an aarch64 host, which only the held GH200 is:
its job shares the node (never `exclusive`, no `gpus`), its CPUs capped (`taskset`) and declared (`cpus`) so other
work keeps the rest, its token staged from bazel.token beside this script (gitignored)."""

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
        {"name": f"PLUGIN-{tag}-x86_64", "kind": "build", "minutes": 120, "secrets": ["DEW_BAZEL_TOKEN"],
         **common, "run": build},
        {"name": f"PLUGIN-{tag}-aarch64", "kind": "program", "gpu": "gh200", "cpus": CORES, "minutes": 120,
         **common, "run": f"DEW_BAZEL_TOKEN_FILE=$SESSION/extra/bazel.token taskset -c 0-{CORES - 1} {build}"},
    ]}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: plugin_jobs.py VERSION SHA")
    json.dump(jobs(sys.argv[1], sys.argv[2]), sys.stdout, indent=1)
    print()
