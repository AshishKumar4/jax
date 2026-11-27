#!/bin/bash
# Dew's JAX CUDA wheels, built from a commit of this fork: the CUDA plugin (jax-cuda12-plugin, jax-cuda13-plugin: its
# kernels and Mosaic GPU's Python extension) for each Python, and the PJRT plugin (jax-cuda12-pjrt, jax-cuda13-pjrt:
# xla_cuda_plugin.so, where Mosaic GPU's custom call compiles a kernel's PTX, its compilation provider among it),
# one a CUDA, as it names no Python. These are the native packages Dew patches: a patch to Mosaic GPU's compile path
# reaches a kernel only through the PJRT plugin (post42 built the CUDA plugin alone, and its kernels still compiled
# through stock PJRT's nvJitLink); and jaxlib (CPU XLA and the Python extensions), each a Python, from the same tree,
# since a patch to XLA (third_party/xla) reaches jax's own lowering and compile paths only through it (post98).
# Bazel builds them with JAX's hermetic C++,
# CUDA and Python toolchains (ci_linux_<machine>_cudaN, as JAX's CI builds its own) and reads and writes Dew's remote
# cache, so a JAX version's cold build is paid once and a patch change rebuilds in minutes.
#
#   build_plugin.sh REF VERSION OUT [PYTHONS] [CUDAS]     on a Linux x86_64 or aarch64 host (its wheels' platform)
#   DEW_PLUGIN_WHEELS=jaxlib DEW_PLUGIN_TESTS_BUILT=@xla//xla/service:hlo_cse_test_interpreter \
#     build_plugin.sh REF VERSION OUT 3.12 12
#   e.g. build_plugin.sh release/0.12.0.1.post44 0.12.0.1.post44 "$OUT"
#
# REF is a branch, a tag or a commit's sha. PYTHONS defaults to every Python Dew supports (its requires-python
# floor to its newest classifier: 3.12, 3.13 and 3.14), so one build is a release's whole set for its machine.
# VERSION is every wheel's version: the commit's jax release (`_version`) and a suffix Bazel and the wheel
# builder both name the wheel with (ML_WHEEL_VERSION_SUFFIX; the commit's own _release_version unset in the
# build tree, so the builder takes the same suffix). DEW_BAZEL_TOKEN_FILE holds the
# cache's bazel-scope token (else DEW_BAZEL_TOKEN, else bazel.token beside this script), which Bazel reads through
# a credential helper when it calls the cache, so no rc file or command line holds it (JAX's .bazelrc announces every
# rc line it reads); DEW_BAZEL_CACHE is the cache's URL. OUT receives the wheels (dist/), their sha256 list
# (wheels.sha256), their objects' debug files by build-id (debug/, debug-<machine>.tar.gz), each build's log and
# seconds (builds.tsv) and the commit built (commit.txt).
set -euo pipefail
ref=$1 version=$2 out=$3 pythons=${4:-"3.12 3.13 3.14"} cudas=${5:-"12 13"}
here=$(cd "$(dirname "$0")" && pwd)
token_file=${DEW_BAZEL_TOKEN_FILE:-$here/bazel.token}
cache=${DEW_BAZEL_CACHE:-https://dew-bazel.ashishkmr472.workers.dev/bazel}
# DEW_PLUGIN_TREE names the build's tree, where a later build on the same node reuses the earlier's Bazel output base
# (keyed by the tree's path) and so rebuilds only the targets its change reaches (post99 on post98, 2026-11-15);
# by default a tree of its own. Its checkout is made afresh either way.
tree=${DEW_PLUGIN_TREE:-$(mktemp -d /tmp/dew-plugin-XXXX)}
mkdir -p "$tree"
rm -rf "$tree/jax"
# On its way out, Bazel's server stopped (its output base kept for the next build): a server left idle is the node's
# user's work to the fleet's watch, which would keep the node up for its idle hours. A server that will not stop
# (Bazel's exit 36, its SIGKILL not taken in time) fails nothing: under errexit it ended the trap and turned a green
# build's exit into 36 (post98 x86_64, 2026-11-15).
trap '(cd "$tree/jax" 2>/dev/null && "${DEW_BAZEL:-bazel}" shutdown) || true
      rm -rf "$tree/jax" "$tree/token" "$tree/credentials"' EXIT
# Where the token is the secret DEW_BAZEL_TOKEN in the environment, it is kept in a file of the build's own for the
# credential helper; a file named explicitly wins.
if [ -z "${DEW_BAZEL_TOKEN_FILE:-}" ] && [ -n "${DEW_BAZEL_TOKEN:-}" ]; then
  token_file="$tree/token"
  (umask 077 && printf %s "$DEW_BAZEL_TOKEN" > "$token_file")
fi
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
# Bazel's credential helper (dew/bazel_credentials.py): the request on stdin, its headers as JSON on stdout, the
# token read from its file when Bazel calls it and on no line Bazel can echo (Dew ledger 6204). The launcher passes
# the helper and the file as quoted arguments.
helper="$tree/credentials"
printf '#!/bin/sh\nexec python3 %q %q "$@"\n' "$here/bazel_credentials.py" "$(realpath "$token_file")" > "$helper"
chmod 700 "$helper"
host=${cache#https://}
host=${host%%/*}
# Read by JAX's .bazelrc (try-import .bazelrc.user), after its own configs. DEW_BAZEL_CACHE=off reads and writes no
# remote cache (row 6204's containment, until its token is rotated); the node's own disk cache (DEW_BAZEL_DISK_CACHE)
# serves either way, needing no token: an action's output is reused only for its exact inputs and command, as the
# remote cache's are, so a second build on the node rebuilds only what its change reaches.
printf 'common --disk_cache=%s\n' "${DEW_BAZEL_DISK_CACHE:-$HOME/.cache/dew-bazel-disk}" > .bazelrc.user
[ "${DEW_BAZEL_CACHE:-}" = off ] || \
  printf 'common --remote_cache=%s\ncommon --credential_helper=%s=%s\ncommon --remote_upload_local_results=true\n' \
    "$cache" "$host" "$helper" >> .bazelrc.user
# DEW_PLUGIN_TESTS: Bazel test targets run in the same tree before the builds, with the first CUDA's config, each
# `target` or `target=FILTER` (its cases alone, --test_filter), every one with --keep_going so one pass reports every
# compile error it reaches; each target's log and Bazel's exit in OUT/tests.tsv (0 passed, 3 a test failed, 1 a
# build failure). The builds run whatever the tests found, so a red test leaves the wheels built; a red test or build
# is the job's failure at its end, every red one's log tail said (Main, 2026-11-15: post98's first compile error
# cost a whole cold build to find).
red=0
if [ -n "${DEW_PLUGIN_TESTS:-}" ]; then
  printf 'target\tseconds\texit\n' > "$out/tests.tsv"
  for test in $DEW_PLUGIN_TESTS; do
    target=${test%%=*} filter=${test#*=} started=$(date +%s) code=0
    log="$out/test-$(echo "$target" | tr '/:@' '___').log"
    "${DEW_BAZEL:-bazel}" test --config=ci_linux_"$(uname -m)"_cuda"${cudas%% *}" \
      --repo_env=HERMETIC_PYTHON_VERSION="${pythons%% *}" "$target" --test_output=all --keep_going \
      ${filter:+$([ "$filter" != "$test" ] && echo "--test_filter=$filter")} > "$log" 2>&1 || code=$?
    printf '%s\t%s\t%s\n' "$test" "$(( $(date +%s) - started ))" "$code" >> "$out/tests.tsv"
    # grep -m, not grep | head: a head that closes its pipe early fails the pipeline under pipefail, and the job.
    [ "$code" = 0 ] || { grep -m 60 -E "error:|FAILED|NO STATUS" "$log" || true; red=$code; }
  done
fi
# DEW_PLUGIN_TESTS_ONLY=1: the tests alone, no wheel built (a gtest's fix rerun on the tree that built the wheels).
if [ "${DEW_PLUGIN_TESTS_ONLY:-}" = 1 ]; then
  exit "$red"
fi
# DEW_PLUGIN_TESTS_BUILT: Bazel test targets built, never run, each target's binary and runfiles kept under
# OUT/tests/<target> with a tarball beside (OUT/tests/<target>.tar.gz) and its build's exit in OUT/tests-built.tsv:
# a build node runs no test (Dew's bible §8), so a target's binary runs where tests run, on the fleet (Main,
# 2026-11-18). Each `target` or `target=FILTER`, the filter only recorded beside it for the run.
keep_built() {
  local target=$1 name=$2 binary
  binary=$("${DEW_BAZEL:-bazel}" cquery --config=ci_linux_"$(uname -m)"_cuda"${cudas%% *}" \
    --repo_env=HERMETIC_PYTHON_VERSION="${pythons%% *}" "$target" --output=starlark \
    --starlark:expr='target.files_to_run.executable.path if target.files_to_run.executable else ""') || return 1
  # Each step checked by its own `|| return`: called as an `if`'s condition, set -e stops nothing inside it.
  [ -n "$binary" ] || { echo "build_plugin: $target names no executable to keep"; return 1; }
  mkdir -p "$out/tests/$name" || return 1
  cp -L "$binary" "$out/tests/$name/" || return 1
  if [ -d "$binary.runfiles" ]; then
    cp -rL "$binary.runfiles" "$out/tests/$name/" || return 1
    rm -f "$out/tests/$name/$(basename "$binary").runfiles/MANIFEST" || return 1
  fi
  tar -C "$out/tests" -czf "$out/tests/$name.tar.gz" "$name" || return 1
}
report=tests.tsv
if [ -n "${DEW_PLUGIN_TESTS_BUILT:-}" ]; then
  report=tests-built.tsv
  printf 'target\tfilter\tseconds\texit\n' > "$out/tests-built.tsv"
  for test in $DEW_PLUGIN_TESTS_BUILT; do
    target=${test%%=*} filter=${test#*=} started=$(date +%s) code=0
    [ "$filter" != "$test" ] || filter=
    name=$(echo "$target" | tr '/:@' '___')
    "${DEW_BAZEL:-bazel}" build --config=ci_linux_"$(uname -m)"_cuda"${cudas%% *}" \
      --repo_env=HERMETIC_PYTHON_VERSION="${pythons%% *}" "$target" --keep_going > "$out/test-$name.log" 2>&1 || code=$?
    # The target's executable, its runfiles beside it with their manifest dropped (its absolute paths are this
    # node's, and a test's runfiles library reads the manifest over the tree), and the pair archived: a step that
    # fails is the target's red build, never the job's exit (set -e), the wheels built after it either way.
    [ "$code" != 0 ] || if ! keep_built "$target" "$name" >> "$out/test-$name.log" 2>&1; then code=1; fi
    printf '%s\t%s\t%s\t%s\n' "$target" "$filter" "$(( $(date +%s) - started ))" "$code" >> "$out/tests-built.tsv"
    [ "$code" = 0 ] || { grep -m 60 -E "error:|ERROR:" "$out/test-$name.log" || true; red=$code; }
  done
fi
# What one cell's build read (row 5670), taken right after it, its repositories as it fetched them, before the next
# cell's options fetch others. Its configured source files by cquery under each wheel's own invocation as build.py
# ran it (its log's [EXECUTING] lines: the options build.py appends itself, --config=cuda_libraries_from_stubs among
# them, beside the `build` options it wrote to .jax_configure.bazelrc, which cquery reads too); the BUILD and .bzl
# files loaded for them (query, which reads no `build` option, given every --repo_env cquery announced, so it
# fetches the cell's repositories and no other); and its repositories' markers and the lockfile
# (`patch_stack.py record`). A query that fails fails the build: a record short of a file would pair a release that
# changed it.
record_cell() {
  local cell="cp${1/./}-cuda$2"
  mkdir -p "$out/read"
  DEW_BAZEL="${DEW_BAZEL:-bazel}" python3 "${DEW_PATCH_STACK:-$here/patch_stack.py}" record "$PWD" \
    "$out/plugin-$cell.log" "$out/read/$cell.log" > "$out/read/$cell.json" ||
    { tail -20 "$out/read/$cell.log"; return 1; }
}
printf 'python\tcuda\tseconds\texit\n' > "$out/builds.tsv"
for python in $pythons; do
  for cuda in $cudas; do
    started=$(date +%s) code=0
    # The PJRT plugin names no Python: built once a CUDA, with the first Python's plugin. jaxlib names no CUDA: built
    # once a Python, with its first CUDA's plugin.
    wheels=jax-cuda-plugin
    [ "$python" != "${pythons%% *}" ] || wheels=$wheels,jax-cuda-pjrt
    [ "$cuda" != "${cudas%% *}" ] || wheels=jaxlib,$wheels
    config=ci_linux_$(uname -m)_cuda$cuda
    # DEW_PLUGIN_WHEELS=jaxlib: jaxlib alone, a CPU change's wheel built without the CUDA plugins (an XLA CPU
    # patch's proof), under JAX's CPU CI config: build.py adds the CUDA configs and hermetic versions the CUDA one
    # assumes only for a CUDA wheel, so jaxlib alone under it failed analysis (2026-11-18, three builds).
    if [ "${DEW_PLUGIN_WHEELS:-}" = jaxlib ]; then
      [ "$cuda" = "${cudas%% *}" ] || continue
      wheels=jaxlib config=ci_linux_$(uname -m)
    fi
    python3 build/build.py build --wheels="$wheels" --cuda_major_version="$cuda" --python_version="$python" \
      --bazel_options=--config="$config" --bazel_options=--repo_env=ML_WHEEL_TYPE=release \
      --bazel_options=--repo_env=ML_WHEEL_VERSION_SUFFIX="$suffix" \
      --bazel_options=--copt=-g1 --bazel_options=--strip=never --bazel_options=--linkopt=-Wl,--build-id \
      --bazel_options=--keep_going --output_path="$out/dist" --verbose > "$out/plugin-cp${python/./}-cuda$cuda.log" 2>&1 || code=$?
    [ "$code" != 0 ] || record_cell "$python" "$cuda" || code=$?
    printf '%s\t%s\t%s\t%s\n' "$python" "$cuda" "$(( $(date +%s) - started ))" "$code" >> "$out/builds.tsv"
    # Each error with the lines after it, where Bazel says why (an analysis failure's "Error in fail: ..." follows
    # its "ERROR: ... in <rule> rule" line, 2026-11-18: two red builds said only that the rule failed).
    [ "$code" = 0 ] || { grep -m 60 -A 6 -E "error:|ERROR:" "$out/plugin-cp${python/./}-cuda$cuda.log" || true
                         echo "build_plugin: a build of $python, CUDA $cuda failed ($code)" >&2; exit "$code"; }
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
# Built with line tables (-g1) and unstripped: each wheel's objects stripped again in place, their debug information
# moved to files named by build-id under OUT/debug (dew/debug_split.py) and packed as debug-<machine>.tar.gz, which
# the release carries beside the wheels, so a stack captured on a host running them reads their frames (row 6013).
# DEW_DEBUG_SPLIT names the splitter where this script is staged apart from it (a fleet job's extra/).
python3 "${DEW_DEBUG_SPLIT:-$here/debug_split.py}" "$out/dist" "$out/debug" | tee "$out/debug.txt"
tar -C "$out/debug" -czf "$out/debug-$(uname -m).tar.gz" .build-id
# What the build read, each cell's (row 5670), merged with the wheels as they are published (after the split
# rewrote them): a later release ships beside these plugins only where it holds every file of every cell's as read
# (`patch_stack.py plugins`). DEW_PATCH_STACK names patch_stack.py where this script is staged apart from it (a
# fleet job's extra/).
python3 "${DEW_PATCH_STACK:-$here/patch_stack.py}" merged "$(cat "$out/commit.txt")" "$out/dist" "$out"/read/*.json \
  > "$out/sources-$(uname -m).json"
(cd "$out/dist" && sha256sum *.whl) | tee "$out/wheels.sha256"
column -t -s $'\t' "$out/builds.tsv"
# A red test leaves the wheels built and fails the job.
[ "$red" = 0 ] || { echo "build_plugin: the wheels are built; a test target was red ($red): OUT/$report" >&2
                    exit "$red"; }
