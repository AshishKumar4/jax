#!/bin/bash
# Dew's JAX CUDA wheels, built from a commit of this fork: the CUDA plugin (jax-cuda12-plugin, jax-cuda13-plugin: its
# kernels and Mosaic GPU's Python extension) for each Python, and the PJRT plugin (jax-cuda12-pjrt, jax-cuda13-pjrt:
# xla_cuda_plugin.so, where Mosaic GPU's custom call compiles a kernel's PTX, its compilation provider among it),
# one a CUDA, as it names no Python. Both are the native packages Dew patches: a patch to Mosaic GPU's compile path
# reaches a kernel only through the PJRT plugin (post42 built the CUDA plugin alone, and its kernels still compiled
# through stock PJRT's nvJitLink); jax is this fork's Python and jaxlib stays the stock release the commit posts.
# Bazel builds them with JAX's hermetic C++,
# CUDA and Python toolchains (ci_linux_<machine>_cudaN, as JAX's CI builds its own) and reads and writes Dew's remote
# cache, so a JAX version's cold build is paid once and a patch change rebuilds in minutes.
#
#   build_plugin.sh REF VERSION OUT [PYTHONS] [CUDAS]     on a Linux x86_64 or aarch64 host (its wheels' platform)
#   e.g. build_plugin.sh release/0.12.0.1.post44 0.12.0.1.post44 "$OUT"
#
# REF is a branch, a tag or a commit's sha. PYTHONS defaults to every Python Dew supports (its requires-python
# floor to its newest classifier: 3.12, 3.13 and 3.14), so one build is a release's whole set for its machine.
# VERSION is every wheel's version: the commit's jax release (`_version`) and a suffix Bazel and the wheel
# builder both name the wheel with (ML_WHEEL_VERSION_SUFFIX; the commit's own _release_version unset in the
# build tree, so the builder takes the same suffix). DEW_BAZEL_TOKEN_FILE holds the
# cache's bazel-scope token (default bazel.token beside this script), which Bazel reads through a credential
# helper when it calls the cache, so no rc file or command line holds it (JAX's .bazelrc announces every rc line
# it reads); DEW_BAZEL_CACHE is the cache's URL. OUT receives the wheels (dist/), their sha256 list
# (wheels.sha256), each build's log and seconds (builds.tsv) and the commit built (commit.txt).
set -euo pipefail
ref=$1 version=$2 out=$3 pythons=${4:-"3.12 3.13 3.14"} cudas=${5:-"12 13"}
here=$(cd "$(dirname "$0")" && pwd)
token_file=${DEW_BAZEL_TOKEN_FILE:-$here/bazel.token}
cache=${DEW_BAZEL_CACHE:-https://dew-bazel.ashishkmr472.workers.dev/bazel}
tree=$(mktemp -d /tmp/dew-plugin-XXXX)
trap 'rm -rf "$tree"' EXIT
mkdir -p "$out/dist"
# Fetched by name or sha alike (git clone --branch takes no sha), then checked out detached.
git init -q "$tree/jax"
cd "$tree/jax"
git fetch -q --filter=blob:none https://github.com/AshishKumar4/jax "$ref"
git checkout -q --detach FETCH_HEAD
git rev-parse HEAD | tee "$out/commit.txt"
base=$(sed -n 's/^_version = "\(.*\)"$/\1/p' jax/version.py)
case "$version" in "$base"?*) suffix=${version#"$base"} ;; *) echo "$version is not $base with a suffix"; exit 1 ;; esac
sed -i "s/^_release_version: str | None = .*/_release_version: str | None = None/" jax/version.py
grep -q "^_release_version: str | None = None$" jax/version.py
# Bazel's credential helper protocol: the request on stdin, its headers as JSON on stdout. The helper reads the
# token file when Bazel calls it.
helper="$tree/credentials"
cat > "$helper" <<EOF
#!/bin/sh
cat > /dev/null
printf '{"headers": {"Authorization": ["Bearer %s"]}}' "\$(tr -d '\n' < '$(realpath "$token_file")')"
EOF
chmod 700 "$helper"
host=${cache#https://}
host=${host%%/*}
# Read by JAX's .bazelrc (try-import .bazelrc.user), after its own configs.
printf 'common --remote_cache=%s\ncommon --credential_helper=%s=%s\ncommon --remote_upload_local_results=true\n' \
  "$cache" "$host" "$helper" > .bazelrc.user
printf 'python\tcuda\tseconds\texit\n' > "$out/builds.tsv"
for python in $pythons; do
  for cuda in $cudas; do
    started=$(date +%s) code=0
    # The PJRT plugin names no Python: built once a CUDA, with the first Python's plugin.
    wheels=jax-cuda-plugin
    [ "$python" != "${pythons%% *}" ] || wheels=jax-cuda-plugin,jax-cuda-pjrt
    python3 build/build.py build --wheels="$wheels" --cuda_major_version="$cuda" --python_version="$python" \
      --bazel_options=--config=ci_linux_"$(uname -m)"_cuda"$cuda" --bazel_options=--repo_env=ML_WHEEL_TYPE=release \
      --bazel_options=--repo_env=ML_WHEEL_VERSION_SUFFIX="$suffix" \
      --output_path="$out/dist" --verbose > "$out/plugin-cp${python/./}-cuda$cuda.log" 2>&1 || code=$?
    printf '%s\t%s\t%s\t%s\n' "$python" "$cuda" "$(( $(date +%s) - started ))" "$code" >> "$out/builds.tsv"
    [ "$code" = 0 ] || { tail -40 "$out/plugin-cp${python/./}-cuda$cuda.log"; exit "$code"; }
  done
done
# Every wheel carries VERSION, or none is published: an OUT reused with an older build's wheels fails here,
# before anything is summed or attached.
for wheel in "$out"/dist/*.whl; do
  case $(basename "$wheel") in
    *-"$version"-*) ;;
    *) echo "build_plugin: $(basename "$wheel") is not $version's; OUT holds another build's wheels" >&2; exit 1 ;;
  esac
done
(cd "$out/dist" && sha256sum *.whl) | tee "$out/wheels.sha256"
column -t -s $'\t' "$out/builds.tsv"
