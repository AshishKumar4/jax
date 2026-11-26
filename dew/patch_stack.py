"""Dew's patch stack on JAX: the fixes Dew carries in this fork until a JAX release has them, moved from one
release to the next.

    python dew/patch_stack.py next              # "OLD NEW": the newest carried release and a newer one
    python dew/patch_stack.py rebase OLD NEW    # in a clone holding jax-vOLD, jax-vNEW and the OLD branch
    python dew/patch_stack.py tests RELEASE     # the stack's own test files, in a clone holding jax-vRELEASE
    python dew/patch_stack.py plugins VERSION   # the plugin release the stack at VERSION ships beside its jax
    python dew/patch_stack.py record WORKSPACE OUTPUT_BASE LOCATIONS   # what a plugin build read (`recorded`)
    python dew/patch_stack.py index MANIFEST SITE FILE...   # the package index, FILEs added to MANIFEST
    python dew/patch_stack.py index MANIFEST SITE --yank FILE... --reason WHY   # FILEs yanked (PEP 592)

`next` reads the releases upstream has tagged (jax-vX.Y.Z) and the stacks this fork carries
(vX.Y.Z-cache-key-pool-fingerprints) and prints the newest stack's release and the newest release, where
that release has no stack yet; nothing where it has.

`rebase` takes the commits of vOLD-cache-key-pool-fingerprints past the tag jax-vOLD and folds them onto jax-vNEW as
vNEW-cache-key-pool-fingerprints, a new branch, so nothing moves backwards. The stack holds each post release's
commits in the order they were released, as dew/release.sh grows each release forward from the last: a carried patch
family's (`[stack] FAMILY: ...`), a repair of one (`fixup! [stack] FAMILY: ...`), and the release commit that sets its
version. The new branch holds one commit a family, then one release commit (row 3689):

- a commit whose change jax-vNEW already holds is dropped: an equal patch (`git cherry`), or one whose fix applies
  as nothing; a family whose every commit is dropped leaves the stack;
- each family's commits are applied in the order they came, each as a three-way merge, and committed as one under
  its first commit's message, author and date, naming every commit it was taken from; a commit named for no family
  is a family of its own. Every commit is made on armada through the fork's own commit hook (RUNNER,
  run_hooked.sh), never in this clone;
- the release commit's version.py is jax-vNEW's with the newest release commit's lines in place of its
  `_release_version` line, the version NEW.post1, so the build sorts after the release it patches and satisfies
  `jax>=NEW`;
- OLD in each message, and in version.py's comment, becomes NEW.

It prints one line a commit (`kept` into its family, `dropped` and why) and one a family committed, and exits 0 with
the branch made; 2 where a commit does not apply, after printing its conflict and leaving no branch; 3 where every fix
was dropped, so only a release commit would remain and the fork has nothing left to carry for NEW.

`tests` prints the test files the stack checked out adds or changes over jax-vRELEASE: each fix's own
regressions, which the workflow runs, with the API they change, before it pushes or publishes a stack.

`plugins` prints the plugin release (plugin-X, its CUDA and PJRT plugins) the stack checked out ships beside its
jax: its own where the fork has one, else an earlier post release's built from the same native code; nothing
where neither is, and its plugins are to be built (dew/build_plugin.sh).

`record`, run by dew/build_plugin.sh after its build, prints what the build read, with digests (`recorded`), which
the plugin release carries as sources-<machine>.json and `plugins` compares a later release against.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import hashlib
import html
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

VERSION = Path("jax/version.py")
RELEASE = re.compile(r'^_release_version: str \| None = "[^"]*"$', re.MULTILINE)
UPSTREAM = "https://github.com/jax-ml/jax.git"
FORK = "https://github.com/AshishKumar4/jax.git"
TAG = re.compile(r"refs/tags/jax-v(\d+(?:\.\d+){2,3})$")
STACK = re.compile(r"refs/heads/v(\d+(?:\.\d+){2,3})-cache-key-pool-fingerprints$")
PLUGINS = re.compile(r"refs/tags/plugin-(\d+(?:\.\d+){2,3})\.post(\d+)$")
FAMILY = re.compile(r"^(?:fixup! )?\[stack\] ([^:]+):")
"""A stack commit's family, from its subject: a family's patch, or a repair of it (`fixup!`)."""
PYTHON_ONLY = ("jax/", "tests/", "docs/", "build/generate_xla_protos.py", ".pre-commit-config.yaml")
"""What a post release changes and still ships plugins built from an earlier one's native code (`plugins`): jax's
Python package, its tests, its docs, build/generate_xla_protos.py, which is run by hand to write jax/_src's
schema modules (xla_protos.py, mosaic_gpu_protos.py) and which no Bazel target, build/build.py or
dew/build_plugin.sh reads (post77's kernel-record schema), and .pre-commit-config.yaml, the commit hooks, which
no build reads (post108's typecheck-clean line, its hooks' config among its files, asked for plugins of its own)."""


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True).stdout


def branch(version: str) -> str:
    return f"v{version}-cache-key-pool-fingerprints"


def tests(release: str) -> list[str]:
    """The test files HEAD adds or changes over jax-vRELEASE: the stack's fixes' own regressions."""
    return git("diff", "--name-only", "--diff-filter=d", f"jax-v{release}", "HEAD", "--", "tests/*_test.py").split()


def native(base: str) -> list[str]:
    """The paths HEAD changes over `base` that a build reads: every one outside `PYTHON_ONLY` (a native source, a
    patch Bazel applies to XLA under third_party/, MODULE.bazel), which reach a user only through Dew's own native
    wheels (post98's XLA patches, beside jaxlib/ and jax_plugins/ alone before)."""
    return git("diff", "--name-only", base, "HEAD", "--", ".", *(f":(exclude){path}" for path in PYTHON_ONLY)).split()


SOURCES = "sources-{machine}.json"
"""A plugin release's record of what its build read on one machine (`recorded`, dew/build_plugin.sh)."""
MACHINES = ("x86_64", "aarch64")
CONFIGURED = (".bazelrc", ".bazelversion", "MODULE.bazel", "platform_mappings", "bazel_downloader.cfg",
              "build/build.py")
"""What configures a build beside the files it reads (`recorded`): Bazel's flags and its version, the modules and
their overrides, and the driver that names its targets and options: a flag or a macro changed over the same sources
builds other code. MODULE.bazel.lock is none: Bazel writes it from MODULE.bazel at a build, and no tree holds it
(it is ignored)."""
STAMP = re.compile(rb'^_release_version: str \| None = (?:"[^"]*"|None)$', re.MULTILINE)
"""A release's version stamp (`RELEASE`), which every release changes and no native code reads as code, as a
release holds it or as a plugin build writes it (None, dew/build_plugin.sh): a digest (`digest`) is of a file
with it blanked."""
LOCATION = re.compile(r"^(.+?):\d+:\d+: ")
"""A path in `bazel query --output=location`'s line."""
LABEL = re.compile(r"^@@([^/]*)//([^:]*):(.+)$")
"""A canonical label (`bazel query --output=label --consistent_labels`'s): its repository (empty: the main
one), package and name."""
MARKED = re.compile(r"^@@([^/]*)//(.*)$")
"""A file a repository rule read, as its repository's marker names it (Bazel's RepoRecordedInput: `@@repo//path`,
a path, not a label; the main repository's `@@//path`), escaped (`unescaped`)."""


def unescaped(text: str) -> str:
    """A marker's field as Bazel wrote it: `\\s` a space, `\\n` a newline, `\\\\` a backslash."""
    return re.sub(r"\\(.)", lambda escape: {"s": " ", "n": "\n"}.get(escape[1], escape[1]), text)


def digest(data: bytes) -> str:
    """The SHA-256 of a file's bytes, its release stamp (`STAMP`) blanked."""
    return hashlib.sha256(STAMP.sub(b'_release_version: str | None = ""', data)).hexdigest()


def overrides(module: str) -> dict[str, dict]:
    """Each archive_override of MODULE.bazel text `module`, by its module_name: its literal arguments."""
    found = {}
    for call in ast.walk(ast.parse(module)):
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "archive_override":
            arguments = {}
            for keyword in call.keywords:
                with contextlib.suppress(ValueError, SyntaxError):
                    arguments[keyword.arg] = ast.literal_eval(keyword.value)
            if "module_name" in arguments:
                found[arguments["module_name"]] = arguments
    return found


def labelled(label: str) -> str:
    """The main repository's path a label names (//third_party/xla:fix.patch: third_party/xla/fix.patch)."""
    package, _, name = label.split("//", 1)[1].partition(":")
    return f"{package}/{name}" if package else name


STARLARK = ("BUILD", "BUILD.bazel", "MODULE.bazel", "REPO.bazel", "WORKSPACE", "WORKSPACE.bazel")
"""Files Bazel evaluates by name (beside every .bzl file): a change to one can change what a build does though no
record names it (a module extension's implementation, whose transitive loads Bazel digests apart), so a release
whose tree changes one a build could have read is paired with no earlier plugins (`differences`)."""
LEGACY_THROUGH = 142
"""The last post release whose plugin builds recorded nothing (`recorded`): a later one's plugins pair a release
only through a record of every machine's build (`built_from`)."""


def starlark(path: str) -> bool:
    """Whether Bazel evaluates `path` (`STARLARK`)."""
    return path.endswith(".bzl") or Path(path).name in STARLARK


def recorded(workspace: Path, output_base: Path, read: list[str]) -> dict:
    """What one plugin build cell read (a Python and a CUDA, as dew/build_plugin.sh builds them), each file's
    `digest`, taken right after the cell's build, its repositories as it fetched them: every file `read` names, by
    its path (`bazel query --output=location`'s lines) or its canonical label (`--output=label
    --consistent_labels`'s, cquery's with its configuration after it: the cell's configured source files, and the
    BUILD and .bzl files loaded for them), every file a repository rule read (each repository's marker in
    OUTPUT_BASE/external) or a module extension did (MODULE.bazel.lock's recorded inputs, written at the build),
    and what configures the build (`CONFIGURED`): the main repository's by path (`main`), each external
    repository's by its canonical name (`external`), Bazel's own (its install base) none. The patches each
    archive_override applies are kept apart (`patches`, by module): their effect is the patched repository's
    files, which a release compares (`differences`)."""
    workspace, external = workspace.resolve(), (output_base / "external").resolve()
    main: dict[str, str] = {}
    outer: dict[str, dict[str, str]] = {}

    def take(path: Path) -> None:
        if not path.is_file():
            return
        real = path.resolve()
        if real.is_relative_to(workspace):
            main[str(real.relative_to(workspace))] = digest(real.read_bytes())
        elif path.is_relative_to(external) and len(path.relative_to(external).parts) > 1:
            repository, *rest = path.relative_to(external).parts
            outer.setdefault(repository, {})[str(Path(*rest))] = digest(real.read_bytes())

    def marked(name: str) -> None:
        # A file as a marker or the lockfile names it: @@repo//path, escaped.
        name = unescaped(name)
        if found := MARKED.match(name):
            take((workspace if not found[1] else external / found[1]) / found[2])
        elif name.startswith("/"):
            take(Path(name))

    for line in read:
        if found := LOCATION.match(line):
            take(Path(found[1]))
        elif found := LABEL.match(line.strip().split(" (", 1)[0]):  # cquery's: its configuration after it.
            take((workspace if not found[1] else external / found[1]) / found[2] / found[3])
    for marker in sorted(external.glob("*.marker")):
        for line in marker.read_text(errors="replace").splitlines():
            if line.startswith("FILE:"):
                marked(line.removeprefix("FILE:").rsplit(" ", 1)[0])
    lock = workspace / "MODULE.bazel.lock"

    def inputs(value) -> None:
        # Every recorded input a module extension read, wherever the lockfile holds it.
        if isinstance(value, dict):
            for key, inner in value.items():
                if isinstance(key, str) and key.startswith("FILE:"):
                    marked(key.removeprefix("FILE:"))
                inputs(inner)
        elif isinstance(value, list):
            for inner in value:
                if isinstance(inner, str) and inner.startswith("FILE:"):
                    marked(inner.removeprefix("FILE:").rsplit(" ", 1)[0])
                inputs(inner)

    if lock.is_file():
        inputs(json.loads(lock.read_text()))
    for name in CONFIGURED:
        take(workspace / name)
    for tool in sorted((workspace / "build" / "tools").rglob("*")):
        take(tool)
    module = (workspace / "MODULE.bazel").read_text() if (workspace / "MODULE.bazel").is_file() else ""
    patches = {}
    for name, override in overrides(module).items():
        paths = [labelled(label) for label in override.get("patches", [])]
        patches[name] = {path: digest((workspace / path).read_bytes()) for path in paths}
        for path in paths:
            main.pop(path, None)
    return {"main": main, "external": outer, "patches": patches}


def merged(cells: dict[str, dict]) -> dict:
    """One machine's record of its build (`SOURCES`): each cell's (`recorded`), by its name, and the patches their
    one tree applies."""
    patches = [cell["patches"] for cell in cells.values()]
    if any(other != patches[0] for other in patches):
        raise ValueError("a build's cells applied different patches")
    return {"cells": {name: {"main": cell["main"], "external": cell["external"]} for name, cell in cells.items()},
            "patches": patches[0] if patches else {}}


def materialized(override: dict, commit: str, into: Path) -> Path:
    """The repository an archive_override makes at `commit` (its archive, verified by its integrity, its
    strip_prefix taken, each of its patches as `commit` holds them applied in order, as Bazel applies them, with
    no fuzz) made under `into`: its root."""
    with urllib.request.urlopen(override["urls"][0], timeout=600) as response:
        data = response.read()
    algorithm, _, expected = override["integrity"].partition("-")
    if hashlib.new(algorithm, data).digest() != base64.b64decode(expected):
        raise ValueError(f"{override['urls'][0]} is not its integrity's archive")
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(into, filter="data")
    root = into / override.get("strip_prefix", "")
    for label in override.get("patches", []):
        patch = into / "patch"
        patch.write_bytes(subprocess.run(["git", "show", f"{commit}:{labelled(label)}"], check=True,
                                         capture_output=True).stdout)
        subprocess.run(["patch", f"-p{override.get('patch_strip', 0)}", "-F0", "-s", "--no-backup-if-mismatch",
                        "-d", str(root), "-i", str(patch)], check=True, capture_output=True)
    return root


def tree(root: Path) -> dict[str, str]:
    """Each file under `root` by its path in it: its `digest`."""
    return {str(path.relative_to(root)): digest(path.read_bytes()) for path in root.rglob("*") if path.is_file()}


def differences(record: dict, built: str, commit: str = "HEAD") -> list[str]:
    """What `commit` holds that could build otherwise than the plugin build at commit `built` did (a machine's
    record, `merged`), each named; none where its native code is the build's. Every cell's main files are compared
    by digest, and a .bzl file `commit` changes over `built` refuses (an extension's code, read by no record). An
    archive_override whose patches `commit` holds as the build did is the same repository (its archive pinned by
    its integrity, MODULE.bazel among the main files); one whose patches differ is made at both commits
    (`materialized`) and refused for any file that differs between them that a cell read, that `commit` adds (a
    glob may take it) or that Bazel evaluates (`starlark`): only a file read by none, unadded and no Starlark, a
    patch's own test changed, pairs; a patch's message changes no file. A repository a module extension generated
    (a canonical name holding "++") is made from inputs the record holds."""
    differing = []
    for name, cell in sorted(record["cells"].items()):
        for path, wanted in sorted(cell["main"].items()):
            held = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True)
            if held.returncode != 0 or digest(held.stdout) != wanted:
                differing.append(f"{name}: {path}")
    changed = git("diff", "--name-only", built, commit).split()
    differing += [f"{path}: Starlark" for path in changed if path.endswith(".bzl")]
    if differing:
        return sorted(set(differing))
    module = subprocess.run(["git", "show", f"{commit}:MODULE.bazel"], capture_output=True, text=True).stdout
    for name, override in overrides(module).items():
        paths = [labelled(label) for label in override.get("patches", [])]
        held = {path: digest(subprocess.run(["git", "show", f"{commit}:{path}"], check=True,
                                            capture_output=True).stdout) for path in paths}
        if record["patches"].get(name) == held:
            continue
        read = {path for cell in record["cells"].values() for repository, files in cell["external"].items()
                if repository in (f"{name}+", f"{name}~") for path in files}
        with tempfile.TemporaryDirectory() as then, tempfile.TemporaryDirectory() as now:
            try:
                before = tree(materialized(override, built, Path(then)))
                after = tree(materialized(override, commit, Path(now)))
            except (OSError, ValueError, subprocess.CalledProcessError, urllib.error.URLError) as failure:
                return [f"{name}'s patched archive: {failure}"]
        for path in sorted(set(before) | set(after)):
            if before.get(path) == after.get(path):
                continue
            why = ("read" if path in read else "added" if path not in before else "Starlark" if starlark(path)
                   else None)
            if why:
                differing.append(f"@{name}//{path}: {why}")
    return differing


def built_from(tag: str) -> dict[str, dict]:
    """The records of what plugin release `tag`'s builds read (`SOURCES`), by each machine it carries one of; none
    for a release built before the builds recorded (`LEGACY_THROUGH`)."""
    found = {}
    for machine in MACHINES:
        address = f"{FORK.removesuffix('.git')}/releases/download/{tag}/{SOURCES.format(machine=machine)}"
        try:
            with urllib.request.urlopen(address, timeout=120) as response:
                found[machine] = json.loads(response.read())
        except urllib.error.HTTPError as failure:
            if failure.code != 404:
                raise
    return found


def plugins(version: str) -> str:
    """The plugin release whose CUDA and PJRT plugins the stack checked out, at `version` (X.postN), ships beside
    its jax: its own, plugin-VERSION, where origin has one; else the newest of an earlier post release of X whose
    native code is HEAD's: HEAD differs only in `PYTHON_ONLY` (jax's Python package, its tests, its docs) from
    the commit its plugins were built at (its tag), or from the stack they were published beside (its release
    branch, release/X.postN, where origin has one), as a post release of Python fixes alone does (Main,
    2026-11-03: post46's two Mosaic GPU fixes; post63's lowering cache, whose Pallas changelog entry is a doc);
    "" where neither is. The published stack counts because a release may take, after its plugins are built, a
    change no build reads (a carried patch's own test, its message), which its own publish pairs with those
    plugins: a later Python-only release then has that stack's native code byte for byte, though not the tag's
    (post82 on post81, 2026-11-12, refused post81's plugins for one patch file's test and header). A dev build's
    (X.postN.devM) is never one: the index never listed it."""
    release, _, post = version.rpartition(".post")
    tags = git("ls-remote", "origin", "refs/tags/plugin-*").split()[1::2]
    published = {int(match.group(2)): tag for tag in tags
                 if (match := PLUGINS.match(tag)) and match.group(1) == release and int(match.group(2)) <= int(post)}
    for found in sorted(published, reverse=True):
        tag = published[found]
        if found == int(post):
            return tag.removeprefix("refs/tags/")
        git("fetch", "-q", "--no-tags", "origin", f"+{tag}:{tag}")
        # A release built since its builds record what they read pairs only through every machine's record, HEAD
        # holding all of each as it was read; an earlier one, by the path rule below.
        records = built_from(tag.removeprefix("refs/tags/"))
        if records or found > LEGACY_THROUGH:
            if set(records) == set(MACHINES) and not any(differences(record, tag) for record in records.values()):
                return tag.removeprefix("refs/tags/")
            continue
        stack = f"refs/remotes/origin/release/{release}.post{found}"
        git("fetch", "-q", "--no-tags", "origin", f"+refs/heads/release/{release}.post{found}:{stack}", check=False)
        held = [tag, *([stack] if git("rev-parse", "-q", "--verify", stack, check=False).strip() else [])]
        if any(not native(base) for base in held):
            return tag.removeprefix("refs/tags/")
    return ""


def upstreamed(old: str, new: str) -> set[str]:
    """The commits of OLD's branch whose patch jax-vNEW holds already (`git cherry`'s `-`)."""
    listed = git("cherry", f"jax-v{new}", branch(old), f"jax-v{old}").split()
    return {sha for mark, sha in zip(listed[::2], listed[1::2], strict=True) if mark == "-"}


def versioned(old: str, new: str, commit: str, post: int) -> str:
    """jax-vNEW's version.py with `commit`'s own lines in place of its `_release_version` line: the lines
    the commit's branch adds to jax-vOLD's (a comment and the version), OLD in them NEW, the version
    NEW.post`post`."""
    added = [line[1:] for line in git("diff", "-U0", f"jax-v{old}", commit, "--", str(VERSION)).splitlines()
             if line.startswith("+") and not line.startswith("+++")]
    lines = "\n".join(added).replace(old, new)
    lines = RELEASE.sub(f'_release_version: str | None = "{new}.post{post}"', lines)
    pristine = git("show", f"jax-v{new}:{VERSION}")
    if pristine.count("\n_release_version: str | None = None\n") != 1:
        raise SystemExit(f"jax-v{new}'s version.py holds no unset _release_version line to set")
    return pristine.replace("\n_release_version: str | None = None\n", f"\n{lines}\n")


def mailed(message: str, author: str, date: str, diff: str) -> str:
    """One commit's patch as `git format-patch -k` writes it (its subject unprefixed, so `[stack]` and `fixup!`
    survive): `diff` under `message`, by `author` at `date`."""
    subject, _, body = message.strip().partition("\n")
    return (f"From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\nFrom: {author}\nDate: {date}\n"
            f"Subject: {subject}\n\n{body.strip()}\n---\n{diff}")


def composed(base: str, label: str, patches: list[str]) -> str | None:
    """`patches` (`mailed`) committed in order onto `base` on armada through the fork's own commit hook (RUNNER,
    run_hooked.sh, taking RUNNER_LOCK, the lane's armada lock), never here: the PC's clone makes no commit (Main,
    2026-11-17). The tip, fetched into this clone; None where the runner refused or did not run, its words said."""
    runner, lock = os.environ.get("RUNNER"), os.environ.get("RUNNER_LOCK")
    if not runner or not lock:
        raise SystemExit("patch_stack: RUNNER names run_hooked.sh, which commits through the fork's hooks on armada, "
                         "and RUNNER_LOCK the lane's armada lock it takes")
    with tempfile.TemporaryDirectory() as scratch:
        files = []
        for index, patch in enumerate(patches):
            files.append(Path(scratch) / f"{index:04}.patch")
            files[-1].write_text(patch)
        clone = git("rev-parse", "--show-toplevel").strip()
        ran = subprocess.run([runner, f"--into={clone}", f"--lock={lock}", label, base, *map(str, files)],
                             capture_output=True, text=True, check=False)
    tip = re.search(r"^tip ([0-9a-f]{40})$", ran.stdout, re.MULTILINE)
    if ran.returncode != 0 or tip is None:
        print(f"the runner did not compose {label} (exit {ran.returncode}):\n{ran.stdout}{ran.stderr}")
        return None
    return tip.group(1)


def rebase(old: str, new: str) -> int:
    commits = git("rev-list", "--reverse", f"jax-v{old}..{branch(old)}").split()
    held = upstreamed(old, new)
    target = branch(new)
    git("checkout", "-q", "--detach", f"jax-v{new}")
    # Each family's patch (`mailed`) the change its commits make, three-way, over the families before it: the index's
    # tree after it against the tree before (`write-tree`, no commit), all committed at the end through the runner.
    base = git("rev-parse", f"jax-v{new}^{{commit}}").strip()
    before, patches = git("write-tree").strip(), []
    # Each family's commits in the order they came, the families in the order each began; the newest release
    # commit, whose version.py lines the new release keeps.
    families: dict[str, list[str]] = {}
    release = None
    for commit in commits:
        paths = git("diff", "--name-only", f"{commit}^", commit).split()
        if str(VERSION) in paths:
            release = commit
            if paths == [str(VERSION)]:
                continue
        named = FAMILY.match(git("log", "-1", "--format=%s", commit))
        families.setdefault(named.group(1) if named else commit, []).append(commit)
    fixes = 0
    for family, members in families.items():
        taken = []
        for commit in members:
            title = git("log", "-1", "--format=%s", commit).strip()
            if commit in held:
                print(f"dropped {commit[:9]} (jax-v{new} holds the same patch): {title}")
                continue
            fix = [path for path in git("diff", "--name-only", f"{commit}^", commit).split() if path != str(VERSION)]
            # The fix applied as a three-way merge, onto the family's commits before it.
            patch = git("diff", "--binary", f"{commit}^", commit, "--", *fix)
            applied = subprocess.run(["git", "apply", "--3way", "--index"], input=patch, capture_output=True,
                                     text=True, check=False)
            if applied.returncode != 0:
                print(f"conflict {commit[:9]}: {title}\n{applied.stderr}{git('diff', check=False)}")
                git("reset", "-q", "--hard")
                return 2
            print(f"kept {commit[:9]} into {family}: {title}")
            taken.append(commit)
        after = git("write-tree").strip()
        if after == before:
            print(f"dropped {family} (applies as nothing on jax-v{new})")
            continue
        first = members[0]
        message = git("log", "-1", "--format=%B", first).replace(old, new)
        author = git("log", "-1", "--format=%an <%ae>", first).strip()
        date = git("log", "-1", "--format=%aD", first).strip()
        patches.append(mailed(f"{message.rstrip()}\n\n(cherry picked from commit {', '.join(taken)})", author, date,
                              git("diff", "--binary", before, after)))
        before = after
        print(f"{family} composed")
        fixes += 1
    if fixes == 0 or release is None:
        print(f"jax-v{new} holds every fix: nothing left to carry" if fixes == 0 else
              f"{branch(old)} holds no release commit to version {new}'s stack by")
        git("checkout", "-q", "--detach", f"jax-v{new}")
        return 3
    VERSION.write_text(versioned(old, new, release, 1))
    git("add", str(VERSION))
    author = git("log", "-1", "--format=%an <%ae>", release).strip()
    date = git("log", "-1", "--format=%aD", release).strip()
    message = f"Release {new}.post1: the stack carried from {old} onto jax-v{new}, a commit a family"
    patches.append(mailed(f"{message}\n\n(cherry picked from commit {release})", author, date,
                          git("diff", "--binary", before, git("write-tree").strip())))
    tip = composed(base, f"rebase-{new}-{int(time.time())}", patches)
    if tip is None:
        git("reset", "-q", "--hard")
        return 4
    git("checkout", "-q", "-f", "-B", target, tip)
    print(f"{target} at {tip}, version {new}.post1")
    return 0


def ordered(version: str) -> tuple[int, ...]:
    """A release's place among releases: 0.12.0.1 (a patch release) after 0.12.0 and before 0.12.1."""
    return tuple(int(part) for part in version.split("."))


def versions(remote: str, pattern: re.Pattern) -> list[str]:
    """The version of each ref of `remote` that `pattern` matches (X.Y.Z, or a patch release's X.Y.Z.W), oldest
    first."""
    listed = git("ls-remote", remote).split()
    return sorted({match.group(1) for match in map(pattern.match, listed[1::2]) if match}, key=ordered)


def following() -> str:
    """"OLD NEW" where the newest release NEW has no stack and OLD is the newest stack's release, else ""."""
    releases, stacks = versions(UPSTREAM, TAG), versions(FORK, STACK)
    newest = releases[-1]
    if not stacks or newest in stacks or ordered(newest) < ordered(stacks[-1]):
        return ""
    return f"{stacks[-1]} {newest}"


def metadata(wheel: Path) -> bytes:
    """A wheel's core metadata, its .dist-info/METADATA (PEP 658)."""
    with zipfile.ZipFile(wheel) as archive:
        (name,) = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        return archive.read(name)


def index(manifest: Path, site: Path, files: list[Path], yank: list[str] = (), reason: str = "",
          package: str = "jax", projects: list[str] = ()) -> None:
    """The index's pages under `site` for `package`, PEP 503's simple API: simple/<package>/index.html, each
    file linked with its digest, its Requires-Python and its metadata's digest (PEP 658, the metadata beside it
    as FILE.metadata), and simple/index.html, which lists `package` and the index's other `projects`, so one
    package's publish keeps the others listed. `manifest` keeps every file of `package` the index has listed, so
    a run adds its own to them; a file it lists already is never replaced, as an installer may hold its
    digest. Each new FILE is copied, with its metadata, under site/packages/<package> and named on stdout.

    Each file named in `yank`, one the manifest lists, is marked yanked for `reason` (PEP 592's
    data-yanked): an installer resolving a range passes it over, and one pinned to exactly it still gets it."""
    listed = json.loads(manifest.read_text()) if manifest.is_file() else {}
    unlisted = sorted(set(yank) - set(listed))
    if unlisted:
        raise SystemExit(f"cannot yank {', '.join(unlisted)}: the index does not list it")
    for name in yank:
        listed[name]["yanked"] = reason
    # A build (X.postN) beside its base release's own wheel (X): an installer that finds the package here looks
    # for every version here (uv's first-index rule), so a lock pinning X stops resolving without it (Main,
    # 2026-10-28: qwix 0.1.8.post1 alone broke every environment main's lock built).
    versions = {name.split("-")[1] for name in [*listed, *(file.name for file in files)]}
    bare = sorted({version.split(".post")[0] for version in versions if ".post" in version} - versions)
    if bare:
        raise SystemExit(f"cannot index {package}'s builds of {', '.join(bare)}: the index lacks those releases' "
                         "own wheels, which an installer finding the package here then cannot resolve; index "
                         "them with it")
    packages = site / "packages" / package
    packages.mkdir(parents=True, exist_ok=True)
    for file in files:
        if file.name in listed:
            print(f"kept {file.name}: listed already", file=sys.stderr)
            continue
        print(file.name)
        (packages / file.name).write_bytes(file.read_bytes())
        core = metadata(file)
        python = next((line.split(":", 1)[1].strip() for line in core.decode().splitlines()
                       if line.startswith("Requires-Python:")), "")
        (packages / f"{file.name}.metadata").write_bytes(core)
        listed[file.name] = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                             "metadata_sha256": hashlib.sha256(core).hexdigest(), "requires_python": python}
    manifest.write_text(json.dumps(listed, indent=1, sort_keys=True) + "\n")
    links = "".join(
        f'<a href="../../packages/{package}/{name}#sha256={entry["sha256"]}"'
        f' data-requires-python="{html.escape(entry["requires_python"])}"'
        f' data-dist-info-metadata="sha256={entry["metadata_sha256"]}"'
        f' data-core-metadata="sha256={entry["metadata_sha256"]}"'
        + (f' data-yanked="{html.escape(entry["yanked"])}"' if "yanked" in entry else "")
        + f'>{name}</a><br>\n'
        for name, entry in sorted(listed.items()))
    (site / "simple" / package).mkdir(parents=True, exist_ok=True)
    root(site, [package, *projects])
    (site / "simple" / package / "index.html").write_text(PAGE.format(title=f"Links for {package}", body=links))


PAGE = ('<!DOCTYPE html>\n<html><head><meta name="pypi:repository-version" content="1.1">'
        "<title>{title}</title></head>\n<body><h1>{title}</h1>\n{body}</body></html>\n")
"""A page of PEP 503's simple API (PEP 629's repository version 1.1: PEP 658's and 714's metadata attributes)."""


STAGED = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin", "jax-cuda13-pjrt")
"""The projects a release stages on its GitHub releases before Dew's index lists them (`indexable`)."""


def canonical(name: str) -> str:
    """A project's name as PEP 503 normalizes it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned(locks: list[str]) -> dict[str, set[str]]:
    """What Dew's locks (`uv pip compile --generate-hashes` files, given as their text) pin of STAGED: each
    `project==version` (its name normalized, its extras dropped), the sha256 of every file the locks allow for it
    (every machine's and Python's)."""
    pins: dict[str, set[str]] = {}
    for text in locks:
        for block in re.split(r"\n(?=[^\s#-])", text):
            found = re.match(r"([A-Za-z0-9_.-]+)(?:\[[^]]*\])?==([^\s;\\]+)", block)
            if found and canonical(found[1]) in STAGED:
                pins.setdefault(f"{canonical(found[1])}=={found[2]}", set()).update(
                    re.findall(r"--hash=sha256:([0-9a-f]{64})", block))
    return pins


def unindexed(pins: dict[str, set[str]], pages: dict[str, str]) -> list[str]:
    """The jax versions Dew's main pins (`pinned`) whose release the index holds not whole: some project and version
    main pins (jax, jaxlib or a CUDA plugin) of which its project's page (`pages`, each project's HTML, none where
    the index has no page for it) lists no wheel. A release whose pin head has passed its train, to be indexed
    (`indexable`) as it lands, never by a hand that remembers (2026-11-17: main pinned post104, then post106, and
    the site's build failed to resolve each until a hand indexed it); one whose index stopped partway, its jax
    listed and a plugin not, among them (review)."""
    missing = False
    for pin in pins:
        project, _, version = pin.partition("==")
        prefix = f"{project.replace('-', '_')}-{version}-"
        listed = re.findall(r">([^<]+\.whl)<", pages.get(project, ""))
        missing = missing or not any(name.startswith(prefix) for name in listed)
    return sorted(pin.partition("==")[2] for pin in pins if pin.startswith("jax==")) if missing else []


def indexable(pins: dict[str, set[str]], wheels: list[Path], version: str) -> list[Path]:
    """Of a release's staged `wheels`, those Dew's main pins (`pinned`), byte for byte, to be indexed: a wheel of a
    pinned project and version whose sha256 the pin does not allow is refused (the bytes differ from what main's train
    ran), as is a set in which main pins no jax of `version` (a pin head that uses the release has not landed); a
    wheel main does not pin is left unindexed."""
    if f"jax=={version}" not in pins:
        raise SystemExit(f"main's locks pin no jax=={version}: its pin head has not landed, so nothing is indexed")
    chosen = []
    for wheel in wheels:
        project, wheel_version = wheel.name.split("-")[:2]
        pin = f"{canonical(project)}=={wheel_version}"
        # A wheel's file name holds its project as the wheel spec normalizes it (lowercase, underscores), which the
        # index step's per-project globs read; another spelling is refused, never left out unsaid (review).
        if pin in pins and project != canonical(project).replace("-", "_"):
            raise SystemExit(f"{wheel.name}: its project is not spelled as a wheel's file name normalizes it")
        if pin not in pins:
            continue
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if digest not in pins[pin]:
            raise SystemExit(f"{wheel.name}'s sha256 {digest} is none main's locks pin for {pin}: refused")
        chosen.append(wheel)
    return chosen


def staged(site: Path, version: str, base: str, manifests: dict[str, dict], assets: list[dict]) -> list[str]:
    """A staging simple index of release `version`, under site/staging/<version>/<digest>/simple/: the fork's own
    projects (STAGED) alone, each with the files Dew's index lists of it (`manifests`, as index() keeps them: linked
    absolutely at `base`, their yanks kept) and the release's staged wheels of it (`assets`: GitHub's {name, digest,
    browser_download_url} of each, from `gh api releases/tags/<tag>`, linked at the asset with GitHub's sha256). A
    lock of the pin head that uses the release resolves those from it, every file of every machine hashed (a
    find-links page gives uv the one wheel it picks: uv #10308), and every other package from /simple, which stays its
    authority (a later release or yank there is seen: Main, 2026-11-16); installs read /simple, which gets nothing
    until that head has landed (`indexable`). `<digest>` names the pages' content, so a page is never rewritten under
    a path a lock names. The pages written, relative to `site`."""
    pages: dict[str, list[str]] = {}
    for project, listed in manifests.items():
        if project not in STAGED:
            continue
        pages[project] = [
            f'<a href="{base}/packages/{project}/{name}#sha256={entry["sha256"]}"'
            f' data-requires-python="{html.escape(entry["requires_python"])}"'
            + core_metadata(entry.get("metadata_sha256"))
            + (f' data-yanked="{html.escape(entry["yanked"])}"' if "yanked" in entry else "")
            + f'>{name}</a><br>\n' for name, entry in sorted(listed.items())]
    # A wheel's core metadata, where its release holds it beside it (attach-plugin.yml's FILE.metadata).
    beside = {found["name"].removesuffix(".metadata"): found.get("digest") or "" for found in assets
              if found["name"].endswith(".whl.metadata")}
    for asset in sorted(assets, key=lambda found: found["name"]):
        name = asset["name"]
        if not name.endswith(".whl"):
            continue
        digest = asset.get("digest") or ""
        if not digest.startswith("sha256:"):
            raise SystemExit(f"{name}: GitHub states no sha256 of it, so no page can name its digest")
        project = canonical(name.split("-")[0])
        if project not in STAGED or name in manifests.get(project, {}):
            continue
        held = beside.get(name, "")
        pages.setdefault(project, []).append(
            f'<a href="{asset["browser_download_url"]}#sha256={digest.removeprefix("sha256:")}"'
            + core_metadata(held.removeprefix("sha256:") if held.startswith("sha256:") else None)
            + f'>{name}</a><br>\n')
    rendered = {project: PAGE.format(title=f"Links for {project}", body="".join(links))
                for project, links in sorted(pages.items())}
    rendered[""] = PAGE.format(title="Simple index",
                               body="".join(f'<a href="{name}/">{name}</a><br>\n' for name in sorted(pages)))
    digest = hashlib.sha256("".join(f"{name}\0{page}\0" for name, page in sorted(rendered.items())).encode())
    prefix = f"staging/{version}/{digest.hexdigest()[:16]}/simple/"
    written = []
    for project, page in rendered.items():
        (site / prefix / project).mkdir(parents=True, exist_ok=True)
        (site / prefix / project / "index.html").write_text(page)
        written.append(prefix + (f"{project}/" if project else ""))
    return sorted(written, key=lambda page: (page == prefix, page))


def core_metadata(digest: str | None) -> str:
    """A link's PEP 658 and 714 attributes for a wheel whose core metadata lies beside it as FILE.metadata, of
    sha256 `digest`; none where it has none, which an installer then reads from the wheel itself: a lock of a
    staging index without them fetched every staged wheel of every machine from GitHub's releases, and 35 at
    once were refused (Dew's train 1273)."""
    if digest is None:
        return ""
    return f' data-dist-info-metadata="sha256={digest}" data-core-metadata="sha256={digest}"'


def root(site: Path, projects: list[str]) -> None:
    """The index's root page, site/simple/index.html, listing each of `projects`."""
    (site / "simple").mkdir(parents=True, exist_ok=True)
    listing = "".join(f'<a href="{name}/">{name}</a><br>\n' for name in sorted(set(projects)))
    (site / "simple" / "index.html").write_text(PAGE.format(title="Simple index", body=listing))


def main() -> int:
    parser = argparse.ArgumentParser(prog="patch_stack.py")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("next")
    rebasing = commands.add_parser("rebase")
    rebasing.add_argument("old")
    rebasing.add_argument("new")
    commands.add_parser("tests").add_argument("release")
    commands.add_parser("plugins").add_argument("version")
    recording = commands.add_parser("record", help="what one plugin build cell read (`recorded`)")
    recording.add_argument("workspace", type=Path)
    recording.add_argument("output_base", type=Path)
    recording.add_argument("read", type=Path, help="the cell's queries' output")
    merging = commands.add_parser("merged", help="a machine's cells' records, as sources-<machine>.json")
    merging.add_argument("cells", type=Path, nargs="+", help="each cell's record, named <cell>.json")
    commands.add_parser("native").add_argument("release")
    indexing = commands.add_parser("index")
    indexing.add_argument("manifest", type=Path)
    indexing.add_argument("site", type=Path)
    indexing.add_argument("files", type=Path, nargs="*")
    indexing.add_argument("--yank", nargs="+", default=[], metavar="FILE", help="the listed files to mark yanked")
    indexing.add_argument("--reason", default="", help="why they are yanked, shown by installers")
    indexing.add_argument("--package", default="jax", help="the package the files and the manifest are of")
    indexing.add_argument("--projects", nargs="*", default=[], help="the index's other packages, for its root")
    pinning = commands.add_parser("pinned", help="what Dew's locks pin of the staged projects, for `indexable`")
    pinning.add_argument("locks", type=Path, nargs="+")
    missing = commands.add_parser("unindexed", help="the jax versions main pins whose release the index lacks")
    missing.add_argument("pins", type=Path, help="`pinned`'s output")
    missing.add_argument("pages", type=Path, help="a directory of <project>.html, the index's page of each")
    choosing = commands.add_parser("indexable", help="the staged wheels main pins, byte for byte")
    choosing.add_argument("pins", type=Path, help="`pinned`'s output")
    choosing.add_argument("version", help="the release's jax version, e.g. 0.12.0.1.post101")
    choosing.add_argument("wheels", type=Path, nargs="*")
    staging = commands.add_parser("staged", help="a release's staging simple index, beside /simple")
    staging.add_argument("site", type=Path)
    staging.add_argument("version")
    staging.add_argument("base", help="the index's address, e.g. https://pypi.dewml.dev")
    staging.add_argument("manifests", type=Path, help="a directory of <project>.json, each project's manifest")
    staging.add_argument("assets", type=Path, nargs="+", help="each release's `gh api releases/tags/<tag>` JSON")
    rooting = commands.add_parser("root")
    rooting.add_argument("site", type=Path)
    rooting.add_argument("--projects", nargs="*", default=[], help="every package the root lists")
    args = parser.parse_args()
    if args.command == "next":
        print(following())
        return 0
    if args.command == "tests":
        print(" ".join(tests(args.release)))
        return 0
    if args.command == "plugins":
        print(plugins(args.version))
        return 0
    if args.command == "record":
        print(json.dumps(recorded(args.workspace, args.output_base, args.read.read_text().splitlines()),
                         sort_keys=True))
        return 0
    if args.command == "merged":
        print(json.dumps(merged({path.stem: json.loads(path.read_text()) for path in args.cells}), sort_keys=True))
        return 0
    if args.command == "native":
        print("\n".join(native(f"jax-v{args.release}")))
        return 0
    if args.command == "pinned":
        print(json.dumps({pin: sorted(digests) for pin, digests in sorted(pinned([lock.read_text() for lock in
                                                                                  args.locks]).items())}))
        return 0
    if args.command == "unindexed":
        pins = {pin: set(digests) for pin, digests in json.loads(args.pins.read_text()).items()}
        pages = {path.stem: path.read_text() for path in sorted(args.pages.glob("*.html"))}
        print("\n".join(unindexed(pins, pages)))
        return 0
    if args.command == "indexable":
        pins = {pin: set(digests) for pin, digests in json.loads(args.pins.read_text()).items()}
        print("\n".join(str(wheel) for wheel in indexable(pins, args.wheels, args.version)))
        return 0
    if args.command == "staged":
        manifests = {path.stem: json.loads(path.read_text()) for path in sorted(args.manifests.glob("*.json"))}
        assets = [asset for path in args.assets for asset in json.loads(path.read_text())["assets"]]
        print("\n".join(staged(args.site, args.version, args.base, manifests, assets)))
        return 0
    if args.command == "root":
        root(args.site, args.projects)
        return 0
    if args.command == "index":
        index(args.manifest, args.site, args.files, args.yank, args.reason, args.package, args.projects)
        return 0
    return rebase(args.old, args.new)


if __name__ == "__main__":
    sys.exit(main())
