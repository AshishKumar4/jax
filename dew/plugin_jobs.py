"""The queue that builds a release's x86_64 CUDA and PJRT plugins (dew/build_plugin.sh) on a build node: a Lambda A10
of role `build`, outside the GPU cap, pushed by hand (runbooks/machines.md, bible section 8), its scripts staged from
beside this script; aarch64's are built by the fork's build-plugin workflow on GitHub's arm64 runners.

    python3 dew/plugin_jobs.py VERSION SHA > dew/plugin-jobs.json
    node launch dew-a10-build-<ts> lambda:gpu_1x_a10 --role build
    node put dew-a10-build-<ts> dew/bazel.token dew/debug_split.py dew/bazel_credentials.py dew/build_plugin.sh
    node push dew-a10-build-<ts> dew/plugin-jobs.json

A plugin build needs no GPU: Bazel's CUDA toolchain is hermetic and Dew's remote cache serves its objects. It never
runs on a GPU node, where its hours of prepare hold the node from lane work (Main, 2026-11-14: post88's x86_64 build,
queued for an A100, kept a prepared exclusive job off that A100's idle GPU for 73 minutes), nor on armada, whose
task disk holds no Bazel output base beside uv's cache (2026-11-10: post74's build ran out of space fetching its
external repositories). The build is the job's prepare; its run lists the wheels the prepare left. The cache's token
is staged from bazel.token beside this script (gitignored)."""

from __future__ import annotations

import json
import re
import sys

FORK = "https://github.com/AshishKumar4/jax"
CORES = 24
"""The build node's cores the build takes, of an A10's 30: Bazel's local actions are few with the remote cache warm."""


def jobs(version: str, sha: str) -> dict:
    """The queue that builds `version`'s x86_64 plugins from the fork's commit `sha` on a build node."""
    # A release's version, or a test build's prerelease of one (`.devN`: no index holds it, a proof installs it by
    # its receipt's paths).
    if not re.fullmatch(r"\d+(\.\d+)+(\.(post|dev)\d+)?", version) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit(f"plugin_jobs: {version!r} is no release version or {sha!r} no full commit sha")
    tag = version.rsplit(".", 1)[-1].upper()
    build = (f"DEW_BAZEL_TOKEN_FILE=$SESSION/extra/bazel.token DEW_DEBUG_SPLIT=$SESSION/extra/debug_split.py "
             f"taskset -c 0-{CORES - 1} bash $SESSION/extra/build_plugin.sh {sha} {version} \"$OUT\"")
    return {"note": f"{version}'s x86_64 plugin build on a build node (dew/plugin_jobs.py)", "inputs": [], "jobs": [
        {"name": f"PLUGIN-{tag}-x86_64", "gpu": "a10", "ref": "main", "keep": [],
         "prepare": f"git ls-remote {FORK} {sha} && {build}", "prepare_minutes": 240, "minutes": 5,
         "run": "ls -l \"$OUT\"/dist/*.whl \"$OUT\"/debug-*.tar.gz",
         "measures": f"{version}'s CUDA and PJRT plugins, every Python and CUDA, for x86_64"}]}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: plugin_jobs.py VERSION SHA")
    json.dump(jobs(sys.argv[1], sys.argv[2]), sys.stdout, indent=1)
    print()
