#!/bin/bash
# Dew's JAX CUDA plugin wheels (jax-cuda12-plugin, jax-cuda13-plugin), built from a commit of this fork: the plugin
# is the one native package Dew patches (its Mosaic GPU compile path and passes); jax is this fork's Python and
# jaxlib and jax-cuda-pjrt stay the stock release the commit posts. Bazel builds them with JAX's hermetic C++,
# CUDA and Python toolchains (ci_linux_x86_64_cudaN, as JAX's CI builds its own) and reads and writes Dew's remote
# cache, so a JAX version's cold build is paid once and a patch change rebuilds in minutes.
#
#   build_plugin.sh REF VERSION OUT [PYTHONS] [CUDAS]     on a Linux x86_64 host
#   e.g. build_plugin.sh plugin-post42 0.12.0.1.post42.dev1 "$OUT" "3.12 3.14" "12 13"
#
# VERSION is every wheel's version: the commit's jax release (`_version`) and a suffix Bazel and the wheel
# builder both name the wheel with (ML_WHEEL_VERSION_SUFFIX; the commit's own _release_version unset in the
# build tree, so the builder takes the same suffix). DEW_BAZEL_TOKEN_FILE holds the
# cache's bazel-scope token (default bazel.token beside this script), which Bazel reads through a credential
# helper when it calls the cache, so no rc file or command line holds it (JAX's .bazelrc announces every rc line
# it reads); DEW_BAZEL_CACHE is the cache's URL. OUT receives the wheels (dist/), their sha256 list
# (wheels.sha256), each build's log and seconds (builds.tsv) and the commit built (commit.txt).
set -euo pipefail
ref=$1 version=$2 out=$3 pythons=${4:-"3.12 3.14"} cudas=${5:-"12 13"}
here=$(cd "$(dirname "$0")" && pwd)
token_file=${DEW_BAZEL_TOKEN_FILE:-$here/bazel.token}
cache=${DEW_BAZEL_CACHE:-https://dew-bazel.ashishkmr472.workers.dev/bazel}
tree=$(mktemp -d /tmp/dew-plugin-XXXX)
trap 'rm -rf "$tree"' EXIT
mkdir -p "$out/dist"
git clone -q --filter=blob:none --branch "$ref" https://github.com/AshishKumar4/jax "$tree/jax"
cd "$tree/jax"
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
    python3 build/build.py build --wheels=jax-cuda-plugin --cuda_major_version="$cuda" --python_version="$python" \
      --bazel_options=--config=ci_linux_x86_64_cuda"$cuda" --bazel_options=--repo_env=ML_WHEEL_TYPE=release \
      --bazel_options=--repo_env=ML_WHEEL_VERSION_SUFFIX="$suffix" \
      --output_path="$out/dist" --verbose > "$out/plugin-cp${python/./}-cuda$cuda.log" 2>&1 || code=$?
    printf '%s\t%s\t%s\t%s\n' "$python" "$cuda" "$(( $(date +%s) - started ))" "$code" >> "$out/builds.tsv"
    [ "$code" = 0 ] || { tail -40 "$out/plugin-cp${python/./}-cuda$cuda.log"; exit "$code"; }
  done
done
(cd "$out/dist" && sha256sum *.whl) | tee "$out/wheels.sha256"
# Every wheel carries VERSION, or none is published.
! ls "$out/dist" | grep '\.whl$' | grep -v -- "-$version-"
column -t -s $'\t' "$out/builds.tsv"
