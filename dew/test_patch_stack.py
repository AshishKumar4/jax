"""The patch stack's choice of the release to carry next, over remotes' refs as git ls-remote lists them, and
the test files it runs, over a repository of its own."""

import base64
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import patch_stack

RELEASED = patch_stack.released
"""`released` itself, which the fixture below replaces for every other test."""


@pytest.fixture(autouse=True)
def unrecorded(monkeypatch):
    """No test reads a plugin release's record from the network: each releases none but its own test's."""
    monkeypatch.setattr(patch_stack, "released", lambda tag: ({}, []))


def listing(*refs: str) -> str:
    return "".join(f"{index:040x}\t{ref}\n" for index, ref in enumerate(refs))


def test_a_patch_release_is_carried_onto_and_one_with_a_stack_is_not(monkeypatch):
    """A patch release (0.12.0.1, four parts) sorts after its release and before the next, is offered as the
    release to carry, and is the newest stack once carried."""
    remotes = {
        patch_stack.UPSTREAM: listing("refs/tags/jax-v0.11.2", "refs/tags/jax-v0.12.0", "refs/tags/jax-v0.12.0.1",
                                      "refs/tags/jax-v0.12.0.1^{}"),
        patch_stack.FORK: listing("refs/heads/dew", "refs/heads/v0.11.2-cache-key-pool-fingerprints",
                                  "refs/heads/v0.12.0-cache-key-pool-fingerprints"),
    }
    monkeypatch.setattr(patch_stack, "git", lambda *args, check=True: remotes[args[1]])
    assert patch_stack.following() == "0.12.0 0.12.0.1"
    remotes[patch_stack.FORK] += listing("refs/heads/v0.12.0.1-cache-key-pool-fingerprints")
    assert patch_stack.following() == ""
    remotes[patch_stack.UPSTREAM] += listing("refs/tags/jax-v0.12.1")
    assert patch_stack.following() == "0.12.0.1 0.12.1"


def wheel(directory, name: str, version: str):
    """A wheel of `name` at `version` holding its metadata alone, as `index` reads it."""
    import zipfile

    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA",
                         f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Python: >=3.10\n")
    return path


def test_each_package_has_its_own_pages_and_the_root_lists_every_one(tmp_path):
    """The index holds Dew's builds of several packages (JAX's, Qwix's): each package's files and pages are its
    own (packages/<name>/, simple/<name>/), and the root page lists every package the index holds, so one's
    publish never drops another's (Main, 2026-10-28: qwix 0.1.8.post1 joins jax)."""
    site = tmp_path / "site"
    jax = wheel(tmp_path, "jax", "0.12.0.1.post31")
    patch_stack.index(tmp_path / "jax.json", site, [wheel(tmp_path, "jax", "0.12.0.1"), jax])
    qwix = wheel(tmp_path, "qwix", "0.1.8.post1")
    patch_stack.index(tmp_path / "qwix.json", site, [wheel(tmp_path, "qwix", "0.1.8"), qwix], package="qwix",
                      projects=["jax"])
    assert (site / "packages" / "qwix" / qwix.name).read_bytes() == qwix.read_bytes()
    assert f'href="../../packages/qwix/{qwix.name}#sha256=' in (site / "simple" / "qwix" / "index.html").read_text()
    assert "qwix" not in (site / "simple" / "jax" / "index.html").read_text()
    root = (site / "simple" / "index.html").read_text()
    assert '<a href="jax/">jax</a>' in root and '<a href="qwix/">qwix</a>' in root
    patch_stack.index(tmp_path / "jax.json", site, [], projects=["qwix"])
    assert '<a href="qwix/">qwix</a>' in (site / "simple" / "index.html").read_text()


def test_the_stacks_tests_are_the_test_files_it_adds_or_changes_over_its_release(tmp_path, monkeypatch):
    """Each test file the stack adds or changes over its release, nested ones among them, and none it leaves
    alone or deletes, nor any file outside tests/ or not a test."""
    def run(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True,
                       capture_output=True)

    def write(files: dict[str, str]) -> None:
        for name, text in files.items():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(text)

    run("init", "-q")
    write({"tests/kept_test.py": "", "tests/changed_test.py": "", "tests/gone_test.py": "",
           "tests/mosaic/gpu_test.py": "", "jax/core.py": ""})
    run("add", "-A")
    run("commit", "-qm", "release")
    run("tag", "jax-v0.12.0.1")
    write({"tests/changed_test.py": "fixed", "tests/mosaic/gpu_test.py": "fixed", "tests/new_test.py": "",
           "tests/conftest.py": "", "jax/core.py": "fixed"})
    (tmp_path / "tests/gone_test.py").unlink()
    run("add", "-A")
    run("commit", "-qm", "the stack")
    monkeypatch.chdir(tmp_path)
    assert patch_stack.tests("0.12.0.1") == ["tests/changed_test.py", "tests/mosaic/gpu_test.py",
                                             "tests/new_test.py"]


def test_a_post_release_ships_its_own_plugins_or_those_built_from_its_native_code(tmp_path, monkeypatch):
    """A post release ships its own plugin release where origin has one; else the newest earlier post release's of
    the same JAX release whose commit it differs from only under jax/, tests/ and docs/; else none, and its plugins are to
    be built. A dev build's, a later post release's and another release's are never one."""
    origin, clone = tmp_path / "origin.git", tmp_path / "clone"

    def run(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=clone, check=True,
                       capture_output=True)

    def commit(files: dict[str, str], message: str) -> None:
        for name, text in files.items():
            (clone / name).parent.mkdir(parents=True, exist_ok=True)
            (clone / name).write_text(text)
        run("add", "-A")
        run("commit", "-qm", message)

    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    clone.mkdir()
    run("init", "-q")
    run("remote", "add", "origin", str(origin))
    commit({"jaxlib/kernel.cc": "post44", "jax/core.py": "post44", "tests/core_test.py": ""}, "post44")
    run("tag", "plugin-0.12.0.1.post44")
    commit({"jaxlib/kernel.cc": "post45"}, "post45")
    run("tag", "plugin-0.12.0.1.post45")
    commit({"jax/core.py": "post46.dev1"}, "post46.dev1")
    run("tag", "plugin-0.12.0.1.post46.dev1")
    run("tag", "plugin-0.12.1.post1")
    run("push", "-q", "origin", "--tags")
    for tag in ("plugin-0.12.0.1.post44", "plugin-0.12.0.1.post45", "plugin-0.12.0.1.post46.dev1",
                "plugin-0.12.1.post1"):
        run("tag", "-d", tag)
    monkeypatch.chdir(clone)

    commit({"jax/core.py": "post46", "jax/version.py": "post46", "tests/core_test.py": "post46",
            "docs/CHANGELOG.md": "post46"}, "post46")
    assert patch_stack.plugins("0.12.0.1.post46") == "plugin-0.12.0.1.post45"
    commit({"jaxlib/kernel.cc": "post46"}, "a native fix")
    assert patch_stack.plugins("0.12.0.1.post46") == ""
    run("tag", "plugin-0.12.0.1.post46")
    run("push", "-q", "origin", "plugin-0.12.0.1.post46")
    assert patch_stack.plugins("0.12.0.1.post46") == "plugin-0.12.0.1.post46"
    run("checkout", "-q", "plugin-0.12.0.1.post45")
    assert patch_stack.plugins("0.12.0.1.post45") == "plugin-0.12.0.1.post45"
    commit({"jax/core.py": "post47"}, "post47")
    assert patch_stack.plugins("0.12.0.1.post47") == "plugin-0.12.0.1.post45"
    # The schema generator, run by hand to write jax/_src's modules and read by no build, changes no plugin; a
    # build script does.
    commit({"build/generate_xla_protos.py": "post48", "build/generate_mosaic_gpu_protos.py": "post48"}, "post48")
    assert patch_stack.plugins("0.12.0.1.post48") == "plugin-0.12.0.1.post45"
    commit({"build/build.py": "post49"}, "post49")
    assert patch_stack.plugins("0.12.0.1.post49") == ""


def test_a_stacks_native_change_is_any_a_build_reads_an_xla_patch_among_them(tmp_path, monkeypatch):
    """`native` lists what a stack changes over its release that a build reads: none for jax's Python, tests and
    docs alone; an XLA patch under third_party/ and MODULE.bazel, which reach a user through Dew's own jaxlib and
    plugins (post98: the publish read jaxlib/ and jax_plugins/ alone and would have shipped XLA's patches as jax
    alone)."""
    clone = tmp_path / "clone"
    clone.mkdir()

    def run(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=clone, check=True,
                       capture_output=True)

    def commit(files: dict[str, str]) -> None:
        for name, text in files.items():
            (clone / name).parent.mkdir(parents=True, exist_ok=True)
            (clone / name).write_text(text)
        run("add", "-A")
        run("commit", "-qm", "a change")

    run("init", "-q")
    commit({"jax/core.py": "release", "MODULE.bazel": "release"})
    run("tag", "jax-v0.12.0.1")
    monkeypatch.chdir(clone)
    commit({"jax/core.py": "a fix", "tests/core_test.py": "its test", "docs/notes.md": "its note",
            ".pre-commit-config.yaml": "its hooks"})
    assert patch_stack.native("jax-v0.12.0.1") == []
    commit({"third_party/xla/early_exit.patch": "the patch", "MODULE.bazel": "lists it"})
    assert sorted(patch_stack.native("jax-v0.12.0.1")) == ["MODULE.bazel", "third_party/xla/early_exit.patch"]


def test_a_python_only_release_ships_the_plugins_of_the_stack_it_grows_from(tmp_path, monkeypatch):
    """A plugin release's wheels were published beside its release's stack (release/X.postN), which may differ
    from the commit the plugins were built at in a file no build reads (a carried patch's own test, its
    message). A later release that differs from THAT stack under jax/, tests/ and docs/ alone has its native
    code, byte for byte, and ships those plugins; one that differs from it anywhere else (a carried patch, a
    native source) ships none until its own are built (0.12.0.1.post82: Python only on post81, whose plugin tag
    differs from post81's stack in one patch file's test and header, was refused the post81 plugins)."""
    origin, clone = tmp_path / "origin.git", tmp_path / "clone"

    def run(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=clone, check=True,
                       capture_output=True)

    def commit(files: dict[str, str], message: str) -> None:
        for name, text in files.items():
            (clone / name).parent.mkdir(parents=True, exist_ok=True)
            (clone / name).write_text(text)
        run("add", "-A")
        run("commit", "-qm", message)

    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    clone.mkdir()
    run("init", "-q")
    run("remote", "add", "origin", str(origin))
    commit({"jaxlib/kernel.cc": "post81", "third_party/xla/fix.patch": "the fix\n", "jax/core.py": "post81"},
           "the commit post81's plugins were built at")
    run("tag", "plugin-0.12.0.1.post81")
    commit({"third_party/xla/fix.patch": "its message, reworded\nthe fix\nits own test\n"}, "post81's release")
    run("push", "-q", "origin", "--tags", "HEAD:refs/heads/release/0.12.0.1.post81")
    run("tag", "-d", "plugin-0.12.0.1.post81")
    monkeypatch.chdir(clone)

    commit({"jax/core.py": "post82", "tests/core_test.py": "post82"}, "post82, Python only")
    assert patch_stack.plugins("0.12.0.1.post82") == "plugin-0.12.0.1.post81"
    commit({"third_party/xla/fix.patch": "its message, reworded\nanother fix\nits own test\n"}, "a patch changed")
    assert patch_stack.plugins("0.12.0.1.post82") == ""
    run("reset", "-q", "--hard", "HEAD~1")
    commit({"jaxlib/kernel.cc": "post82"}, "a native source changed")
    assert patch_stack.plugins("0.12.0.1.post82") == ""


def test_a_plugin_build_cell_records_every_file_it_read_and_what_configures_it(tmp_path):
    """A plugin build cell's record (`recorded`) holds each file its queries listed (a source file by its location
    or by cquery's label, its configuration after it; a loaded file by its canonical label), each a repository rule
    read (its repository's marker: an escaped path) or a module extension did (the lockfile's recorded inputs), the
    main repository's by path and an external one's by its canonical name, with Bazel's flags, its modules and its
    driver; a version stamp blanked, as a release or the build tree writes it; an archive_override's patches
    apart, by module; nothing else of the tree. A machine's record holds its cells by name, the commit it built and
    each wheel it made by its sha256 (`merged`, row 5670), and is refused without them."""
    workspace, output_base = tmp_path / "jax", tmp_path / "output_base"
    files = {"jaxlib/kernel.cc": "int k;\n", "jax/version.py": "_release_version: str | None = None\n",
             ".bazelrc": "build --copt=-O2\n", "build/build.py": "TARGETS = 1\n", "docs/notes.md": "no build\n",
             "third_party/xla/fix.patch": "the fix\n", "build/requirements.txt": "numpy\n",
             "MODULE.bazel": 'archive_override(module_name = "xla", patches = ["//third_party/xla:fix.patch"])\n'}
    for name, text in files.items():
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_text(text)
    # A build's checkout, as dew/build_plugin.sh clones it: the drivers it tracks are what a record takes of them.
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True)
    for name in ("xla/compiled.cc", "xla/BUILD", "third_party/gpus/cuda.tpl", "third_party/gpus/a b.json"):
        (output_base / "external" / "xla+" / name).parent.mkdir(parents=True, exist_ok=True)
        (output_base / "external" / "xla+" / name).write_text(name)
    (output_base / "external" / "@@xla+.marker").write_text(
        "FILE:@@xla+//third_party/gpus/cuda.tpl 0a\nFILE:@@xla+//third_party/gpus/a\\sb.json 0c\n"
        "FILE:@@//third_party/xla/fix.patch 0b\nENV:CUDA 12\n")
    (workspace / "MODULE.bazel.lock").write_text(json.dumps({"moduleExtensions": {"@@//build:pip.bzl%pip": {
        "general": {"bzlTransitiveDigest": "x", "recordedInputs": {"FILE:@@//build/requirements.txt": "0d"}}}}}))
    read = [f"{workspace}/jax/version.py:1:1: source file //jax:version.py",
            "@@//jaxlib:kernel.cc (null)", "@@xla+//xla:compiled.cc (5a1b2c)", "@@xla+//xla:BUILD",
            "@@bazel_tools//tools/cpp:BUILD"]
    record = patch_stack.recorded(workspace, output_base, read)
    assert sorted(record["main"]) == [".bazelrc", "MODULE.bazel", "build/build.py", "build/requirements.txt",
                                      "jax/version.py", "jaxlib/kernel.cc"]
    for stamped in (b'_release_version: str | None = "0.12.0.1.post82"\n', b"_release_version: str | None = None\n"):
        assert record["main"]["jax/version.py"] == patch_stack.digest(stamped)
    assert record["external"] == {"xla+": {name: patch_stack.digest(name.encode()) for name in (
        "xla/compiled.cc", "xla/BUILD", "third_party/gpus/cuda.tpl", "third_party/gpus/a b.json")}}
    assert record["patches"] == {"xla": {"third_party/xla/fix.patch": patch_stack.digest(b"the fix\n")}}
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "jaxlib-1-cp312.whl").write_bytes(b"a wheel")
    wheels = sorted((tmp_path / "dist").glob("*.whl"))
    machine = patch_stack.merged({"cp312-cuda12": record, "cp312-cuda13": record}, "c" * 40, wheels)
    assert sorted(machine["cells"]) == ["cp312-cuda12", "cp312-cuda13"] and machine["patches"] == record["patches"]
    assert machine["commit"] == "c" * 40
    assert machine["wheels"] == {"jaxlib-1-cp312.whl": hashlib.sha256(b"a wheel").hexdigest()}
    for commit, made in (("v0.12", wheels), ("c" * 40, [])):
        with pytest.raises(ValueError):
            patch_stack.merged({"cp312-cuda12": record}, commit, made)


def test_a_release_ships_the_plugins_whose_builds_read_nothing_it_changed(tmp_path, monkeypatch):
    """Where an earlier plugin release recorded what each machine's build read, a later release ships its plugins
    only where every machine's record is there and it holds each cell's files as read (`differences`); a patch to
    XLA is made at both commits from its pinned archive and refused for a file that differs between them that a
    cell read, that it adds or that Bazel evaluates: the patch's own test changed and its message reworded pair; a
    changed compiled line, an added file, a .bzl the patch or the tree changes, a native source, or a flag over the
    same sources (.bazelrc) does not; one machine's record alone pairs nothing; a release built since the records
    began (past LEGACY_THROUGH) pairs through its records alone (row 5670: post81's pairing judged by hand). It is
    compared at the commit its records name, never its tag's (an attach tags a version, at whatever commit); a
    BUILD.bazel beside a package a cell read refuses, a test package's BUILD pairs; and a record pairs only beside
    the wheels it describes, its own version's release too (a partial attach, then a rebuild's record)."""
    archive_root = tmp_path / "archive" / "xla-1"
    (archive_root / "xla").mkdir(parents=True)
    (archive_root / "xla" / "compiled.cc").write_text("int f() { return 1; }\n")
    (archive_root / "xla" / "BUILD").write_text("cc_library(name = 'compiled')\n")
    (archive_root / "xla" / "defs.bzl").write_text("X = 1\n")
    tarball = tmp_path / "xla-1.tar.gz"
    subprocess.run(["tar", "-czf", str(tarball), "-C", str(tmp_path / "archive"), "xla-1"], check=True)
    integrity = "sha256-" + base64.b64encode(hashlib.sha256(tarball.read_bytes()).digest()).decode()
    module = (f'archive_override(module_name = "xla", integrity = "{integrity}", patch_strip = 1,\n'
              f'    patches = ["//third_party/xla:fix.patch"], strip_prefix = "xla-1",\n'
              f'    urls = ["{tarball.as_uri()}"])\n')

    def fix(line: str, test: str = "TEST(F, One) {}", message: str = "the fix\n", extra: str = "") -> str:
        return (f"{message}--- a/xla/compiled.cc\n+++ b/xla/compiled.cc\n@@ -1 +1 @@\n"
                f"-int f() {{ return 1; }}\n+{line}\n"
                f"--- /dev/null\n+++ b/xla/compiled_test.cc\n@@ -0,0 +1 @@\n+{test}\n{extra}")

    origin, clone = tmp_path / "origin.git", tmp_path / "clone"

    def run(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=clone, check=True,
                       capture_output=True)

    def commit(files: dict[str, str], message: str) -> None:
        for name, text in files.items():
            (clone / name).parent.mkdir(parents=True, exist_ok=True)
            (clone / name).write_text(text)
        run("add", "-A")
        run("commit", "-qm", message)

    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    subprocess.run(["git", "-C", str(origin), "config", "uploadpack.allowAnySHA1InWant", "true"], check=True)
    clone.mkdir()
    run("init", "-q", "-b", "main")
    run("remote", "add", "origin", str(origin))
    # A compiled source the checkout holds as a link to one of two implementations.
    (clone / "jaxlib").mkdir()
    for impl in ("a", "b"):
        (clone / "jaxlib" / f"impl_{impl}.cc").write_text(f"int {impl};\n")
    (clone / "jaxlib" / "target.cc").symlink_to("impl_a.cc")
    commit({"jaxlib/kernel.cc": "int k;\n", "jax/core.py": "post81", ".bazelrc": "build --copt=-O2\n",
            "jax/version.py": '_release_version: str | None = "0.12.0.1.post81"\n', "MODULE.bazel": module,
            "third_party/xla/fix.patch": fix("int f() { return 2; }")}, "the commit post81's plugins were built at")
    built = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, check=True, capture_output=True,
                           text=True).stdout.strip()
    run("push", "-q", "origin", "HEAD:refs/heads/main")
    # Its tags at another commit, as an attach's tag may be (a tooling commit of the fork's): compared there, its
    # .bzl would refuse every release.
    run("checkout", "-q", "-b", "tooling")
    commit({"jaxlib/tooling.bzl": "Z = 1\n"}, "a tooling commit the tags were made at")
    for tag in ("plugin-0.12.0.1.post81", "plugin-0.12.0.1.post143", "plugin-0.12.9.post1"):
        run("tag", tag)
    run("push", "-q", "origin", "--tags")
    for tag in ("plugin-0.12.0.1.post81", "plugin-0.12.0.1.post143", "plugin-0.12.9.post1"):
        run("tag", "-d", tag)
    run("checkout", "-q", "main")
    monkeypatch.chdir(clone)
    # The build's record: its tree as the build stamps it, XLA as its override makes it, what its queries listed.
    output_base = tmp_path / "output_base"
    (output_base / "external").mkdir(parents=True)
    xla = patch_stack.materialized(patch_stack.overrides(module)["xla"], "HEAD", tmp_path / "made")
    (output_base / "external" / "xla+").symlink_to(xla)
    (clone / "jax" / "version.py").write_text("_release_version: str | None = None\n")
    cell = patch_stack.recorded(clone, output_base, [
        "@@//jaxlib:kernel.cc (null)", "@@//jaxlib:target.cc (null)", "@@//jax:version.py (null)", "@@xla+//xla:compiled.cc (5a1b2c)",
        "@@xla+//xla:BUILD"])
    run("checkout", "--", "jax/version.py")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "jaxlib-0.12.0.1.post81-cp312.whl").write_bytes(b"post81's jaxlib")
    record = patch_stack.merged({"cp312-cuda12": cell}, built, sorted((tmp_path / "dist").glob("*.whl")))
    built = [("x86_64", record), ("aarch64", record)]
    uploaded = dict(record["wheels"])
    monkeypatch.setattr(patch_stack, "released",
                        lambda tag: (uploaded, built) if tag == "plugin-0.12.0.1.post81" else ({}, []))

    commit({"jax/core.py": "post82", "tests/core_test.py": "post82",
            "jax/version.py": '_release_version: str | None = "0.12.0.1.post82"\n'}, "post82, Python only")
    verified: dict[str, str] = {}
    assert patch_stack.plugins("0.12.0.1.post82", verified=verified) == "plugin-0.12.0.1.post81"
    assert verified == record["wheels"]
    commit({"third_party/xla/fix.patch": fix("int f() { return 2; }", test="TEST(F, Two) {}",
                                             message="its message, reworded\n")}, "the patch's own test")
    # Each patched archive made once a commit, however many machines' records name it.
    making, made = patch_stack.materialized, []
    monkeypatch.setattr(patch_stack, "materialized", lambda *args: made.append(args[1]) or making(*args))
    assert patch_stack.plugins("0.12.0.1.post82") == "plugin-0.12.0.1.post81"
    assert sorted(made) == sorted([built[0][1]["commit"], "HEAD"])
    monkeypatch.setattr(patch_stack, "materialized", making)
    # A test package's BUILD, which no cell read a file of, pairs.
    commit({"tests/BUILD": "py_test(name = 'core_test')\n"}, "a test target")
    assert patch_stack.plugins("0.12.0.1.post82") == "plugin-0.12.0.1.post81"
    added = "--- /dev/null\n+++ b/xla/new.cc\n@@ -0,0 +1 @@\n+int g;\n"
    starlark = "--- a/xla/defs.bzl\n+++ b/xla/defs.bzl\n@@ -1 +1 @@\n-X = 1\n+X = 2\n"
    for change in ({"third_party/xla/fix.patch": fix("int f() { return 3; }")},
                   {"third_party/xla/fix.patch": fix("int f() { return 2; }", extra=added)},
                   {"third_party/xla/fix.patch": fix("int f() { return 2; }", extra=starlark)},
                   {"jaxlib/kernel.cc": "int k2;\n"}, {".bazelrc": "build --copt=-O3\n"},
                   {"jaxlib/rules.bzl": "Y = 1\n"}, {"jaxlib/BUILD.bazel": "cc_library(name = 'kernel')\n"},
                   {"BUILD.bazel": "package()\n"}, {"build/tools/utils/__init__.py": "shadowing = True\n"},
                   {".bazeliskrc": "USE_BAZEL_VERSION=8.4.2\n"}):
        commit(change, "a change a build could read")
        assert patch_stack.plugins("0.12.0.1.post82") == ""
        run("reset", "-q", "--hard", "HEAD~1")
    # A link retargeted to the other implementation, neither file changed: what the build reads changed.
    (clone / "jaxlib" / "target.cc").unlink()
    (clone / "jaxlib" / "target.cc").symlink_to("impl_b.cc")
    run("add", "-A")
    run("commit", "-qm", "the link retargeted")
    assert patch_stack.plugins("0.12.0.1.post82") == ""
    run("reset", "-q", "--hard", "HEAD~1")
    # Records beside other wheels than theirs (an attach's record of a rebuild, its first build's wheels kept)
    # pair nothing.
    uploaded["jaxlib-0.12.0.1.post81-cp312.whl"] = hashlib.sha256(b"another build's jaxlib").hexdigest()
    assert patch_stack.plugins("0.12.0.1.post82") == ""
    uploaded.update(record["wheels"])
    # A wheel the release holds that no record describes pairs nothing.
    uploaded["jax_cuda12_plugin-0.12.0.1.post81-cp313.whl"] = "f" * 64
    assert patch_stack.plugins("0.12.0.1.post82") == ""
    uploaded.pop("jax_cuda12_plugin-0.12.0.1.post81-cp313.whl")
    # A second build of one machine for other Pythons has its own record: the two cover the release's wheels
    # together, each bound to its own; one of them bound to no wheel the release holds pairs nothing.
    later = {**record, "wheels": {"jaxlib-0.12.0.1.post81-cp313.whl": hashlib.sha256(b"cp313").hexdigest()}}
    uploaded.update(later["wheels"])
    built.append(("x86_64", later))
    assert patch_stack.plugins("0.12.0.1.post82") == "plugin-0.12.0.1.post81"
    uploaded.pop("jaxlib-0.12.0.1.post81-cp313.whl")
    assert patch_stack.plugins("0.12.0.1.post82") == ""
    built.pop()
    # One machine's record alone pairs nothing, but for a publish of that machine's plugins alone.
    built.pop()
    assert patch_stack.plugins("0.12.0.1.post82") == ""
    assert patch_stack.plugins("0.12.0.1.post82", ("x86_64",)) == "plugin-0.12.0.1.post81"
    # A release past LEGACY_THROUGH pairs through its records alone, never by the path rule.
    assert patch_stack.plugins("0.12.0.1.post144") == ""
    # A series LEGACY_THROUGH names no post of records from its first: its release with none pairs nothing.
    assert patch_stack.plugins("0.12.9.post1") == ""
    monkeypatch.setattr(patch_stack, "released", lambda tag: (uploaded, [("x86_64", record), ("aarch64", record)]))
    assert patch_stack.plugins("0.12.0.1.post144") == "plugin-0.12.0.1.post143"
    # A release that cannot be read whole (its listing, or a record it lists, gone) pairs nothing; the search goes on.
    def unread(tag):
        raise patch_stack.Unread(f"{tag}: 404")
    monkeypatch.setattr(patch_stack, "released", unread)
    assert patch_stack.plugins("0.12.0.1.post144") == ""
    monkeypatch.setattr(patch_stack, "released", lambda tag: (uploaded, [("x86_64", record), ("aarch64", record)]))
    # A release lacking a wheel the publish ships (`lacking`: a first batch of Pythons alone) pairs nothing.
    monkeypatch.setattr(patch_stack, "lacking", lambda *args: ["jax-cuda12-plugin cp313 x86_64"])
    assert patch_stack.plugins("0.12.0.1.post144", matrix=(("cp313",), ("12",))) == ""
    monkeypatch.setattr(patch_stack, "lacking", lambda *args: [])
    assert patch_stack.plugins("0.12.0.1.post144", matrix=(("cp313",), ("12",))) == "plugin-0.12.0.1.post143"
    # Its own version's release ships beside its records only where they describe its wheels and HEAD holds what
    # they read: a compiled change since its build is no build of it.
    assert patch_stack.plugins("0.12.0.1.post143") == "plugin-0.12.0.1.post143"
    commit({"jaxlib/kernel.cc": "int k2;\n"}, "a compiled change since the build")
    assert patch_stack.plugins("0.12.0.1.post143") == ""
    run("reset", "-q", "--hard", "HEAD~1")
    uploaded.clear()
    assert patch_stack.plugins("0.12.0.1.post143") == ""


def test_a_release_lacks_each_wheel_a_publish_ships_that_it_holds_none_of():
    """`lacking`: each Python's jaxlib (where the release holds a jaxlib at all) and CUDA plugin, and each CUDA's
    PJRT plugin, for each machine a publish ships; a first batch of Pythons alone lacks the rest."""
    version, machine = "0.12.1.post3", "x86_64"
    whole = {f"jaxlib-{version}-cp312-cp312-manylinux_2_27_{machine}.whl",
             f"jaxlib-{version}-cp313-cp313-manylinux_2_27_{machine}.whl",
             f"jax_cuda12_pjrt-{version}-py3-none-manylinux_2_27_{machine}.whl",
             f"jax_cuda12_plugin-{version}-cp312-cp312-manylinux_2_27_{machine}.whl",
             f"jax_cuda12_plugin-{version}-cp313-cp313-manylinux_2_27_{machine}.whl"}
    matrix = (("cp312", "cp313"), ("12",))
    assert patch_stack.lacking(whole, version, (machine,), *matrix) == []
    first = {name for name in whole if "cp313" not in name}
    assert patch_stack.lacking(first, version, (machine,), *matrix) == [
        "jaxlib cp313 x86_64", "jax-cuda12-plugin cp313 x86_64"]
    stock = {name for name in whole if not name.startswith("jaxlib")}
    assert patch_stack.lacking(stock, version, (machine,), *matrix) == []
    assert patch_stack.lacking(whole, version, (machine, "aarch64"), *matrix) == [
        "jaxlib cp312 aarch64", "jaxlib cp313 aarch64", "jax-cuda12-pjrt aarch64",
        "jax-cuda12-plugin cp312 aarch64", "jax-cuda12-plugin cp313 aarch64"]


def test_a_releases_failed_uploads_are_none_of_its_files(monkeypatch):
    """`released` reads a release's uploaded files alone: a failed upload's placeholder (GitHub's state "starter"),
    a wheel or a record, is neither a file it holds nor a record it fetches."""
    url = "https://github.com/AshishKumar4/jax/releases/download/plugin-0.12.1.post3"
    listing = {"assets": [
        {"name": "jaxlib-0.12.1.post3-cp312.whl", "state": "uploaded", "digest": "sha256:" + "1" * 64,
         "browser_download_url": f"{url}/jaxlib.whl"},
        {"name": "jaxlib-0.12.1.post3-cp313.whl", "state": "starter", "digest": None,
         "browser_download_url": f"{url}/partial.whl"},
        {"name": "sources-x86_64-py3.13.json", "state": "starter", "digest": None,
         "browser_download_url": f"{url}/sources.json"}]}

    def opened(address, timeout):
        assert address.startswith("https://api.github.com/"), f"{address}: a placeholder fetched"
        return io.BytesIO(json.dumps(listing).encode())
    monkeypatch.setattr(patch_stack.urllib.request, "urlopen", opened)
    assert RELEASED("plugin-0.12.1.post3") == ({"jaxlib-0.12.1.post3-cp312.whl": "1" * 64}, [])


def test_a_build_is_refused_where_the_index_lacks_its_base_release(tmp_path):
    """A fork's build (0.1.8.post1) is indexed only beside its base release's own wheel (0.1.8), listed already or
    indexed with it: an installer that finds the package on this index looks for every version here (uv's
    first-index rule), so a lock pinning the base would stop resolving (Main, 2026-10-28: qwix 0.1.8.post1 alone
    broke every environment main's lock built)."""
    site = tmp_path / "site"
    qwix = wheel(tmp_path, "qwix", "0.1.8.post1")
    try:
        patch_stack.index(tmp_path / "qwix.json", site, [qwix], package="qwix")
    except SystemExit as refused:
        assert "0.1.8" in str(refused) and not (tmp_path / "qwix.json").exists()
    else:
        raise AssertionError("a build without its base release was indexed")
    patch_stack.index(tmp_path / "qwix.json", site, [wheel(tmp_path, "qwix", "0.1.8")], package="qwix")
    patch_stack.index(tmp_path / "qwix.json", site, [qwix], package="qwix")
    assert qwix.name in (site / "simple" / "qwix" / "index.html").read_text()



GH = """#!/usr/bin/env python3
# gh as release.sh calls it: a dispatch recorded with its token, the run named for it listed (gh having applied
# the caller's --jq) from the second listing after it, unless the fork says its runs never appear, and that run's
# conclusion success, unless the fork says its stage goes red.
import json, os, sys
from pathlib import Path
state = Path(__file__).with_name("gh.json")
calls = json.loads(state.read_text()) if state.exists() else []
calls.append(sys.argv[1:])
state.write_text(json.dumps(calls))
if sys.argv[1:3] == ["run", "list"] and not os.environ.get("RUNS_NEVER_APPEAR"):
    query = sys.argv[sys.argv.index("--jq") + 1]
    dispatched = [call for call in calls if call[:2] == ["workflow", "run"]]
    listings = sum(call[:2] == ["run", "list"] for call in calls)
    token = next(arg for arg in dispatched[-1] if arg.startswith("token=")) if dispatched else ""
    if dispatched and listings >= 2 and token.removeprefix("token=") in query:
        print(7)
if sys.argv[1:3] == ["run", "view"]:
    print(os.environ.get("RUN_ENDED") or ("completed failure" if os.environ.get("STAGE_RED") else "completed success"))
if sys.argv[1] == "api":
    if os.environ.get("STAGED"):
        sys.exit(0)
    sys.exit(print("gh: Server Error (HTTP 502)" if os.environ.get("LOOKUP_FAILS") else "gh: Not Found (HTTP 404)",
                   file=sys.stderr) or 1)
"""


# run_hooked.sh as release.sh calls it, composing where armada would: BASE's tree, each `pick:SHA` cherry-picked and
# each patch file applied as its commit (`git am -k`), the tip fetched into --into as refs/hooked/LABEL; a series
# that does not apply exits 2, as the runner does, having fetched nothing.
RUNNER = """#!/bin/bash
set -euo pipefail
into=${1#--into=}; shift
[[ $1 == --lock=* ]] && shift
label=$1 base=$2; shift 2
work=$(mktemp -d)
git -C "$into" worktree add -q --detach "$work" "$base"
for entry in "$@"; do
  case $entry in
    pick:*) git -C "$work" cherry-pick "${entry#pick:}" > /dev/null 2>&1 || exit 2 ;;
    *) git -C "$work" am -q -k "$entry" > /dev/null 2>&1 || exit 2 ;;
  esac
done
tip=$(git -C "$work" rev-parse HEAD)
git -C "$into" update-ref "refs/hooked/$label" "$tip"
echo "tip $tip"
"""
# as_patch.sh beside the runner: a diff on stdin as one commit's patch under the message file's message.
AS_PATCH = """#!/bin/bash
set -euo pipefail
printf 'From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\\nFrom: t <t@example.com>\\n'
printf 'Date: %s\\nSubject: %s\\n\\n' "$(date -R)" "$(head -n 1 "$1")"
tail -n +2 "$1" | sed '1{/^$/d;}'
printf -- '---\\n'
cat
"""


def isolated(tmp_path) -> dict:
    """An environment whose git reads no configuration but its own and installs no hooks or templates, and
    finds gh and sleep in tmp_path/bin first, then git where this machine has it."""
    (tmp_path / "templates").mkdir()
    git = Path(shutil.which("git") or "/usr/bin/git").parent
    return {"PATH": f"{tmp_path / 'bin'}:{git}:/usr/bin:/bin", "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TEMPLATE_DIR": str(tmp_path / "templates"),
            **{f"GIT_{who}_{what}": value for who in ("AUTHOR", "COMMITTER")
               for what, value in (("NAME", "t"), ("EMAIL", "t@example.com"))}}


def fork(tmp_path, environment):
    """A fork whose release/0.12.0.1.post39 and stack branch hold one commit, a clone of it, and branches of
    the clone's: a family's fix to carry, two families that conflict with each other, and a commit of no family
    that sets a near-miss version."""
    def run(cwd, *args):
        return subprocess.run(["git", "-C", str(cwd), *args], env=environment, check=True, capture_output=True,
                              text=True).stdout.strip()

    origin, checkout = tmp_path / "origin.git", tmp_path / "checkout"
    run(tmp_path, "init", "-q", "--bare", str(origin))
    run(tmp_path, "init", "-q", "-b", "main", str(checkout))
    (checkout / "jax").mkdir()
    (checkout / "jax" / "version.py").write_text('_release_version: str | None = "0.12.0.1.post39"\n')
    (checkout / "lax.py").write_text("one\n")
    run(checkout, "add", "-A")
    run(checkout, "commit", "-qm", "post39")
    run(checkout, "remote", "add", "origin", str(origin))
    run(checkout, "push", "-q", "origin", "HEAD:refs/heads/release/0.12.0.1.post39",
        "HEAD:refs/heads/v0.12.0.1-cache-key-pool-fingerprints")
    commits = {}
    for name, path, text in (("fix", "fix.py", "fixed\n"), ("other", "lax.py", "two\n"),
                             ("conflicting", "lax.py", "three\n"),
                             ("near", "jax/version.py", '_release_version: str | None = "0.12.0.1+post40"\n')):
        run(checkout, "checkout", "-q", "-b", name, "main")
        (checkout / path).write_text(text)
        run(checkout, "add", "-A")
        run(checkout, "commit", "-qm", name if name == "near" else f"[stack] {name}: {name}")
        commits[name] = run(checkout, "rev-parse", "HEAD")
        # On the fork, as every commit a release picks is (the runner fetches it there).
        run(checkout, "push", "-q", "origin", f"{name}:refs/heads/{name}")
    return origin, checkout, commits, run


def test_a_release_is_pushed_and_published_only_once_every_step_before_it_has_held(tmp_path):
    """release.sh pushes the release branch, asks patch-stack.yml to publish from it and watches the run its
    dispatch's token names, and moves the stack branch forward to it only once that run has passed, the release
    grown on top of the one before so both move only forward, all only once every commit it carries has applied
    as a family's and the version reads exactly the new release: a cherry-pick that conflicts, a commit of no
    family, or a stack branch moved off the release it grows from, ends the run with nothing pushed and nothing
    asked. A run that goes red or never appears leaves the stack branch where it was, the composition kept by a
    tag and its release branch removed, so the release is composed again with nothing moved back (2026-11-17:
    post105's red stage left the stack branch at it, and moving it back rewrote pushed history)."""
    environment = isolated(tmp_path)
    origin, checkout, commits, run = fork(tmp_path, environment)
    (tmp_path / "bin").mkdir()
    for name, text in (("gh", GH), ("sleep", "#!/bin/sh\n"), ("run_hooked.sh", RUNNER), ("as_patch.sh", AS_PATCH)):
        (tmp_path / "bin" / name).write_text(text)
        (tmp_path / "bin" / name).chmod(0o755)
    environment = {**environment, "RUNNER": str(tmp_path / "bin" / "run_hooked.sh"),
                   "RUNNER_LOCK": str(tmp_path / "armada.lock")}
    calls = tmp_path / "bin" / "gh.json"
    script = Path(patch_stack.__file__).with_name("release.sh")

    def released(new, *picked, **extra):
        return subprocess.run(["bash", str(script), str(checkout), "0.12.0.1.post39", new, f"Release {new}",
                               *picked], env={**environment, **extra}, capture_output=True, text=True,
                              check=False)

    def branches():
        return run(origin, "for-each-ref", "--format=%(refname) %(objectname)")

    before = branches()
    # Each of these runs as the clean one below does but for the commits it carries.
    for picked in ((commits["other"], commits["conflicting"]), (commits["near"],)):
        assert released("0.12.0.1.post40", *picked).returncode != 0
        assert branches() == before and not calls.exists()
    # Wheels built at a commit (BUILT_AT) whose source the composition does not equal but for its version, or a
    # message that names it not, push and ask nothing (FarSilverfish, 2026-11-17: a native release built ahead).
    base = run(origin, "rev-parse", "release/0.12.0.1.post39")
    for built, said in ((base, f"Release 0.12.0.1.post40, its plugins built at {base[:10]}"),
                        (commits["fix"], "Release 0.12.0.1.post40"),
                        ("0" * 40, f"Release 0.12.0.1.post40, its plugins built at {'0' * 10}")):
        refused = subprocess.run(["bash", str(script), str(checkout), "0.12.0.1.post39", "0.12.0.1.post40", said,
                                  commits["fix"]], env={**environment, "BUILT_AT": built}, capture_output=True,
                                 text=True, check=False)
        assert refused.returncode != 0 and "BUILT_AT" in refused.stderr
        assert branches() == before and not calls.exists()
    # Wheels built at a commit whose source differs from the composition only in what no native wheel is built from
    # (a test, the jax package's Python): the release goes on (2026-11-18: post111's test skip on its built source).
    run(checkout, "checkout", "-q", "-b", "built", base)
    run(checkout, "cherry-pick", commits["fix"])
    (checkout / "tests").mkdir(exist_ok=True)
    (checkout / "tests" / "t.py").write_text("skipped before")
    run(checkout, "add", "-A")
    run(checkout, "commit", "-qm", "built")
    built = run(checkout, "rev-parse", "HEAD")
    run(checkout, "push", "-q", "origin", f"{built}:refs/heads/built")
    done = subprocess.run(["bash", str(script), str(checkout), "0.12.0.1.post39", "0.12.0.1.post40",
                           f"Release 0.12.0.1.post40, its plugins built at {built[:10]}", commits["fix"]],
                          env={**environment, "BUILT_AT": built}, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    stack = run(origin, "rev-parse", "v0.12.0.1-cache-key-pool-fingerprints")
    assert run(origin, "rev-parse", "release/0.12.0.1.post40") == stack
    assert run(origin, "show", f"{stack}:jax/version.py") == '_release_version: str | None = "0.12.0.1.post40"'
    assert run(origin, "show", f"{stack}:fix.py") == "fixed"
    published = run(origin, "rev-parse", "release/0.12.0.1.post39")
    assert run(origin, "rev-list", f"{published}..{stack}").split() == [stack, run(origin, "rev-parse", f"{stack}^")]
    assert run(origin, "log", "-1", "--format=%s", f"{stack}^") == "[stack] fix: fix"
    asked = json.loads(calls.read_text())
    dispatch, = [call for call in asked if call[:2] == ["workflow", "run"]]
    assert dispatch[:9] == ["workflow", "run", "patch-stack.yml", "-R", "AshishKumar4/jax", "--ref", "dew", "-f",
                            "publish=0.12.0.1"]
    assert "from=release/0.12.0.1.post40" in dispatch
    assert ["run", "watch", "7", "-R", "AshishKumar4/jax"] in asked
    # The stack branch has moved off post39: a release grown from post39 again is refused, unasked.
    before, count = branches(), len(asked)
    assert released("0.12.0.1.post41", commits["fix"]).returncode != 0
    assert branches() == before and len(json.loads(calls.read_text())) == count
    # A red stage, then a run that never appears: the stack branch stays at post40 each time, the composition kept
    # by a tag and its release branch removed, and post41 is composed again over it, nothing moved back.
    def again(new, *picked, **extra):
        return subprocess.run(["bash", str(script), str(checkout), "0.12.0.1.post40", new, f"Release {new}",
                               *picked], env={**environment, **extra}, capture_output=True, text=True,
                              check=False)

    # A run whose watch stopped before its end, a red one that staged its wheels first, and a red one whose staging
    # could not be read (an error other than GitHub's 404): each left as it is,
    # its release branch kept for the run to go on or be run again, the stack branch where it was.
    for how in ({"RUN_ENDED": "in_progress "}, {"STAGE_RED": "1", "STAGED": "1"}, {"STAGE_RED": "1", "LOOKUP_FAILS": "1"}):
        assert again("0.12.0.1.post41", commits["other"], **how).returncode != 0
        assert run(origin, "rev-parse", "v0.12.0.1-cache-key-pool-fingerprints") == stack
        assert "release/0.12.0.1.post41" in branches() and not run(origin, "tag", "--list", "failed/*")
        run(origin, "update-ref", "-d", "refs/heads/release/0.12.0.1.post41")
    for how in ({"STAGE_RED": "1"}, {"RUNS_NEVER_APPEAR": "1"}):
        assert again("0.12.0.1.post41", commits["other"], **how).returncode != 0
        assert run(origin, "rev-parse", "v0.12.0.1-cache-key-pool-fingerprints") == stack
        assert "release/0.12.0.1.post41" not in branches()
        kept = run(origin, "tag", "--list", "failed/0.12.0.1.post41-*").split()
        assert kept and all(run(origin, "show", f"{tag}:lax.py") == "two" for tag in kept)
    assert again("0.12.0.1.post41", commits["other"]).returncode == 0
    grown = run(origin, "rev-parse", "v0.12.0.1-cache-key-pool-fingerprints")
    assert run(origin, "rev-parse", "release/0.12.0.1.post41") == grown
    assert run(origin, "rev-parse", f"{grown}^^") == stack


def test_a_rebase_folds_the_stack_into_a_commit_a_family_and_one_release(tmp_path, monkeypatch):
    """`rebase` carries OLD's stack onto jax-vNEW as one commit a family, each family's repairs (`fixup!`) folded
    into it in order, a family jax-vNEW holds dropped, and one release commit versioned NEW.post1 last, however many
    post releases OLD's stack grew through (row 3689)."""
    def run(*args: str) -> str:
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True,
                              capture_output=True, text=True).stdout

    def commit(files: dict[str, str], message: str) -> None:
        for name, text in files.items():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(text)
        run("add", "-A")
        run("commit", "-qm", message)

    unset = "_version: str = '0.12.0'\n_release_version: str | None = None\n"
    run("init", "-q", "-b", "main")
    commit({"jax/version.py": unset, "jax/a.py": "a\n", "jax/b.py": "b\n"}, "jax 0.12.0")
    run("tag", "jax-v0.12.0")
    run("checkout", "-q", "-b", "v0.12.0-cache-key-pool-fingerprints")
    commit({"jax/a.py": "a fixed\n"}, "[stack] fix-a: a is fixed")
    commit({"jax/version.py": unset.replace("None = None", 'None = "0.12.0.post1"')}, "Release 0.12.0.post1: fix-a")
    commit({"jax/b.py": "b fixed\n"}, "[stack] fix-b: b is fixed")
    commit({"jax/a.py": "a fixed again\n"}, "fixup! [stack] fix-a: a is fixed")
    commit({"jax/version.py": unset.replace("None = None", 'None = "0.12.0.post2"')}, "Release 0.12.0.post2: fix-b")
    run("checkout", "-q", "main")
    commit({"jax/version.py": unset.replace("0.12.0", "0.12.1"), "jax/b.py": "b fixed\n"}, "jax 0.12.1: b fixed")
    run("tag", "jax-v0.12.1")
    monkeypatch.chdir(tmp_path)
    # A committer, as the workflow's runner configures one: a machine with no git identity (an armada vessel) has none.
    for who, value in (("NAME", "t"), ("EMAIL", "t@t")):
        monkeypatch.setenv(f"GIT_COMMITTER_{who}", value)
    # Its commits made where armada would, by the runner's stand-in.
    (tmp_path.parent / "run_hooked.sh").write_text(RUNNER)
    (tmp_path.parent / "run_hooked.sh").chmod(0o755)
    monkeypatch.setenv("RUNNER", str(tmp_path.parent / "run_hooked.sh"))
    monkeypatch.setenv("RUNNER_LOCK", str(tmp_path.parent / "armada.lock"))
    assert patch_stack.rebase("0.12.0", "0.12.1") == 0
    subjects = run("log", "--format=%s", "jax-v0.12.1..v0.12.1-cache-key-pool-fingerprints").splitlines()
    assert subjects == ["Release 0.12.1.post1: the stack carried from 0.12.0 onto jax-v0.12.1, a commit a family",
                        "[stack] fix-a: a is fixed"]
    assert run("show", "v0.12.1-cache-key-pool-fingerprints:jax/a.py") == "a fixed again\n"
    assert '_release_version: str | None = "0.12.1.post1"' in run(
        "show", "v0.12.1-cache-key-pool-fingerprints:jax/version.py")


def test_a_release_is_indexed_only_as_mains_locks_pin_it_byte_for_byte(tmp_path):
    """What patch-stack's index takes of a staged release: the wheels Dew's main pins (`pinned` of its locks), each
    whose sha256 the pin allows; a pinned wheel whose bytes differ is refused, and nothing is indexed where main
    pins no jax of the release (its pin head has not landed) (Main, 2026-11-15: post98 and post99 were indexed
    before any Dew test ran on them)."""
    jax = wheel(tmp_path, "jax", "0.12.0.1.post101")
    jaxlib = wheel(tmp_path, "jaxlib", "0.12.0.1.post101")
    unpinned = wheel(tmp_path, "jax_cuda13_plugin", "0.12.0.1.post101")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    lock = (f"jax[cuda12]==0.12.0.1.post101 ; sys_platform == 'linux' \\\n"
            f"    --hash=sha256:{digest(jax)}\n    # via dewml\n"
            f"jaxlib==0.12.0.1.post101 \\\n    --hash=sha256:{'0' * 64} \\\n    --hash=sha256:{digest(jaxlib)}\n"
            f"numpy==2.3.0 \\\n    --hash=sha256:{'1' * 64}\n")
    pins = patch_stack.pinned([lock])
    assert set(pins) == {"jax==0.12.0.1.post101", "jaxlib==0.12.0.1.post101"}
    assert patch_stack.indexable(pins, [jax, jaxlib, unpinned], "0.12.0.1.post101") == [jax, jaxlib]
    other = tmp_path / "other"
    other.mkdir()
    changed = wheel(other, "jaxlib", "0.12.0.1.post101")
    with zipfile.ZipFile(changed, "a") as archive:
        archive.writestr("changed", "bytes")
    with pytest.raises(SystemExit):
        patch_stack.indexable(pins, [jax, changed], "0.12.0.1.post101")
    with pytest.raises(SystemExit):
        patch_stack.indexable(pins, [jax], "0.12.0.1.post102")
    # Indexed as it lands: the jax version main pins while any project and version it pins is on no page of the
    # index (a version only a longer one's prefix matches is not one); its jax listed and its jaxlib not, a release
    # indexed partway, still asked; all listed, not asked again.
    def page(name: str) -> str:
        return f'<a href="https://x/{name}#sha256=0">{name}</a><br>\n'

    jaxed = page("jax-0.12.0.1.post10-py3-none-any.whl")
    assert patch_stack.unindexed(pins, {"jax": jaxed}) == ["0.12.0.1.post101"]
    jaxed += page("jax-0.12.0.1.post101-py3-none-any.whl")
    assert patch_stack.unindexed(pins, {"jax": jaxed}) == ["0.12.0.1.post101"]
    whole = {"jax": jaxed, "jaxlib": page("jaxlib-0.12.0.1.post101-cp312-cp312-manylinux_2_27_x86_64.whl")}
    assert patch_stack.unindexed(pins, whole) == []



def test_a_staging_index_states_each_wheels_core_metadata_where_it_lies_beside_it(tmp_path):
    """A staging page (`staged`) states a file's core metadata (PEP 658 and 714) where it lies beside it: an indexed
    file's at /packages (its manifest's digest), a release's wheel's as its release's FILE.metadata asset (GitHub's
    sha256 of it); a wheel with none beside it is linked without, which an installer reads from the wheel itself."""
    manifests = {"jaxlib": {"jaxlib-0.12.0.1-cp312-cp312-manylinux_2_27_x86_64.whl": {
        "sha256": "1" * 64, "metadata_sha256": "2" * 64, "requires_python": ">=3.12"}}}
    url = "https://github.com/AshishKumar4/jax/releases/download/plugin-0.12.0.1.post101"
    held, bare = (f"jaxlib-0.12.0.1.post101-cp312-cp312-manylinux_2_27_{machine}.whl"
                  for machine in ("x86_64", "aarch64"))
    assets = [{"name": name, "digest": "sha256:" + digit * 64, "browser_download_url": f"{url}/{name}"}
              for name, digit in ((held, "3"), (f"{held}.metadata", "4"), (bare, "5"))]
    prefix = patch_stack.staged(tmp_path, "0.12.0.1.post101", "https://pypi.dewml.dev", manifests, assets)[-1]
    links = {found[1]: found[0] for found in re.findall(r'(<a href="[^"]*/([^/"#]+)#[^>]*>)',
                                                        (tmp_path / prefix / "jaxlib" / "index.html").read_text())}
    indexed = links["jaxlib-0.12.0.1-cp312-cp312-manylinux_2_27_x86_64.whl"]
    assert 'data-core-metadata="sha256=' + "2" * 64 + '"' in indexed
    assert 'data-core-metadata="sha256=' + "4" * 64 + '"' in links[held]
    assert "data-core-metadata" not in links[bare] and "data-dist-info-metadata" not in links[bare]
    assert f"{held}.metadata" not in links


def test_a_staging_index_stages_the_plugin_wheels_its_records_bound_for_the_machines_shipped(tmp_path):
    """A staging index's plugin wheels (`staged`'s `plugin`) are the shipped release's wheels for the machines the
    publish ships, and where its builds' records bound them (`verified`, as the publish downloaded them), exactly
    those: an asset uploaded since the pairing read the release is never staged, and one the release no longer
    holds at its bound digest refuses; a release whose builds recorded nothing stages every wheel of a machine."""
    url = "https://github.com/AshishKumar4/jax/releases/download/plugin-0.12.1.post3"

    def wheel(name, digit):
        return {"name": name, "digest": "sha256:" + digit * 64, "browser_download_url": f"{url}/{name}"}
    x86, arm, late = (
        "jaxlib-0.12.1.post3-cp312-cp312-manylinux_2_27_x86_64.whl",
        "jaxlib-0.12.1.post3-cp312-cp312-manylinux_2_27_aarch64.whl",
        "jax_cuda12_plugin-0.12.1.post3-cp312-cp312-manylinux_2_27_x86_64.whl")
    plugin = [wheel(x86, "3"), wheel(arm, "4"), wheel(late, "9"), wheel("debug-x86_64-py3.12.tar.gz", "5")]
    verified = {x86: "3" * 64, arm: "4" * 64}

    def staged(into, verified, machines):
        prefix = patch_stack.staged(tmp_path / into, "0.12.1.post3", "https://pypi.dewml.dev", {}, [], plugin,
                                    verified, machines)[-1]
        return {name: (tmp_path / into / prefix / name / "index.html").read_text()
                for name in ("jaxlib", "jax-cuda12-plugin") if (tmp_path / into / prefix / name).is_dir()}
    both = staged("both", verified, ("x86_64", "aarch64"))
    assert x86 in both["jaxlib"] and arm in both["jaxlib"] and "jax-cuda12-plugin" not in both
    alone = staged("alone", verified, ("x86_64",))
    assert x86 in alone["jaxlib"] and arm not in alone["jaxlib"]
    legacy = staged("legacy", {}, ("x86_64",))
    assert x86 in legacy["jaxlib"] and late in legacy["jax-cuda12-plugin"] and arm not in legacy["jaxlib"]
    with pytest.raises(SystemExit, match="no longer holds"):
        staged("moved", {**verified, x86: "8" * 64}, ("x86_64", "aarch64"))


def test_a_staging_index_lists_the_forks_projects_alone_every_file_hashed(tmp_path):
    """A release's staging simple index (`staged`): the fork's own projects alone, each with the files the index lists
    of it (linked absolutely, their yanks kept) and the release's staged wheels at their GitHub assets with GitHub's
    sha256, every machine's (a lock resolving from it hashes them all: uv #10308); every other package left to /simple
    (a later release or yank there is seen); its path named by its content, so a rendering of other pages never lands
    under one a lock names; a non-wheel asset left out; an asset GitHub states no sha256 of refused."""
    manifests = {"jaxlib": {"jaxlib-0.12.0.1-cp312-cp312-manylinux_2_27_x86_64.whl": {
        "sha256": "1" * 64, "metadata_sha256": "2" * 64, "requires_python": ">=3.12", "yanked": "why"}},
        "dewml": {"dewml-0.2.0-py3-none-any.whl": {"sha256": "7" * 64, "metadata_sha256": "8" * 64,
                                                    "requires_python": ">=3.12"}}}
    url = "https://github.com/AshishKumar4/jax/releases/download"
    assets = [{"name": f"jaxlib-0.12.0.1.post101-cp312-cp312-manylinux_2_27_{machine}.whl",
               "digest": f"sha256:{d * 64}",
               "browser_download_url": f"{url}/plugin-0.12.0.1.post101/jaxlib-{machine}.whl"}
              for machine, d in (("x86_64", "3"), ("aarch64", "4"))]
    assets += [{"name": "debug-x86_64.tar.gz", "digest": "sha256:" + "5" * 64, "browser_download_url": f"{url}/d"},
               {"name": "jax-0.12.0.1.post101-py3-none-any.whl", "digest": "sha256:" + "6" * 64,
                "browser_download_url": f"{url}/jax-0.12.0.1.post101/jax.whl"}]
    written = patch_stack.staged(tmp_path, "0.12.0.1.post101", "https://pypi.dewml.dev", manifests, assets)
    prefix = written[-1]
    assert prefix.startswith("staging/0.12.0.1.post101/") and prefix.endswith("/simple/")
    simple = tmp_path / prefix
    jaxlib = (simple / "jaxlib" / "index.html").read_text()
    assert 'href="https://pypi.dewml.dev/packages/jaxlib/jaxlib-0.12.0.1-cp312' in jaxlib
    assert 'data-yanked="why"' in jaxlib
    assert "#sha256=" + "3" * 64 in jaxlib and "#sha256=" + "4" * 64 in jaxlib and "debug" not in jaxlib
    root = (simple / "index.html").read_text()
    assert 'href="jax/"' in root and 'href="jaxlib/"' in root and "dewml" not in root
    assert not (simple / "dewml").exists()
    again = patch_stack.staged(tmp_path / "again", "0.12.0.1.post101", "https://pypi.dewml.dev", manifests, assets)
    other = patch_stack.staged(tmp_path / "other", "0.12.0.1.post101", "https://pypi.dewml.dev", {}, assets)
    assert again[-1] == prefix and other[-1] != prefix
    with pytest.raises(SystemExit):
        patch_stack.staged(tmp_path, "0.12.0.1.post102", "https://pypi.dewml.dev", manifests,
                           [{**assets[0], "digest": None}])


def test_fixes_lists_the_manifests_fixes_by_topic_and_status():
    """`fixes` lists the manifest's fixes in its topics' order, those of a topic or a status alone where named, and
    refuses a topic or status the manifest has none of, naming those it has."""
    manifest = {"topics": [{"topic": "xla", "title": "X"}, {"topic": "pallas", "title": "P"}], "fixes": [
        {"fix": "b", "topic": "pallas", "status": "carried", "deleted_when": "upstream #2"},
        {"fix": "a", "topic": "pallas", "status": "retired", "deleted_when": "none stated"},
        {"fix": "c", "topic": "xla", "status": "carried", "deleted_when": "XLA 1234"}]}
    assert patch_stack.fixes(manifest) == [
        "xla carried c: deleted when XLA 1234", "pallas carried b: deleted when upstream #2",
        "pallas retired a: deleted when none stated"]
    assert patch_stack.fixes(manifest, topic="pallas", status="carried") == [
        "pallas carried b: deleted when upstream #2"]
    with pytest.raises(SystemExit, match="no topic 'mosaic' in the manifest: pallas, xla"):
        patch_stack.fixes(manifest, topic="mosaic")
    with pytest.raises(SystemExit, match="no status 'moved'"):
        patch_stack.fixes(manifest, status="moved")


def test_the_manifest_states_every_fix_in_one_of_its_topics():
    """The stack's own manifest: each fix in one of its topics, with a status and an exit stated."""
    manifest = json.loads(patch_stack.MANIFEST.read_text())
    topics = {entry["topic"] for entry in manifest["topics"]}
    assert len(topics) == len(manifest["topics"])
    for fix in manifest["fixes"]:
        assert fix["topic"] in topics, fix["fix"]
        assert fix["status"] in ("carried", "reworked", "moved", "retired", "withdrawn"), fix["fix"]
        assert fix["deleted_when"], fix["fix"]


def test_each_cells_record_queries_under_its_wheels_own_build_options():
    """A cell's record queries under each wheel build's own options as build.py ran it (its log's [EXECUTING]
    lines), --config=cuda_libraries_from_stubs, which it appends itself, among them: queried under the options it
    writes alone, the CI config's include_cuda_libs=true failed the wheels' analysis (row 5670)."""
    log = (
        "INFO building\n"
        "[EXECUTING] /usr/bin/bazelisk build --config=ci_linux_x86_64_cuda --repo_env=HERMETIC_CUDA_VERSION=12.9.1 "
        "--config=cuda_libraries_from_stubs '--action_env=PATH=/a b' //jaxlib/tools:jaxlib_wheel\n"
        "noise [EXECUTING] /usr/bin/bazelisk build --config=cuda_libraries_from_stubs "
        "//jaxlib/tools:jax_cuda_plugin_wheel\n"
        "[EXECUTING] /usr/bin/bazelisk info output_base\n")
    found = patch_stack.invocations(log)
    assert [target for _, _, target in found] == [
        "//jaxlib/tools:jaxlib_wheel", "//jaxlib/tools:jax_cuda_plugin_wheel"]
    assert {bazel for bazel, _, _ in found} == {"/usr/bin/bazelisk"}
    assert found[0][1] == ["--config=ci_linux_x86_64_cuda", "--repo_env=HERMETIC_CUDA_VERSION=12.9.1",
                           "--config=cuda_libraries_from_stubs", "--action_env=PATH=/a b"]
    assert all("--config=cuda_libraries_from_stubs" in options for _, options, _ in found)
    # Each option Bazel announced, a line at a time: one ending its line takes no word of the next.
    assert patch_stack.announced_envs("INFO: --repo_env=A=1\n--repo_env B=2 --config=x\n") == [
        "--repo_env=A=1", "--repo_env=B=2"]


def test_a_cells_reads_are_queried_with_each_option_its_own_argument(tmp_path):
    """`read_by` runs Bazel itself, each option its own argument as the build's log gave it (no shell splits one
    holding a space): cquery under each wheel build's options, then query over every target given each --repo_env
    cquery announced; Bazel's own output goes to the errors file, and a query that fails fails the record."""
    calls = tmp_path / "calls.jsonl"
    bazel = tmp_path / "bazel"
    bazel.write_text(f"""#!{sys.executable}
import json, sys
open({str(calls)!r}, "a").write(json.dumps(sys.argv[1:]) + "\\n")
if "--fail" in sys.argv:
    sys.exit(2)
if sys.argv[1] == "cquery":
    print("--repo_env=HERMETIC_CUDA_VERSION=12.9.1 announced", file=sys.stderr)
    print("@@//jaxlib:kernel.cc (null)")
else:
    print("@@//jaxlib:BUILD")
""")
    bazel.chmod(0o755)
    log = (f"[EXECUTING] {bazel} build --config=x '--action_env=PATH=/a b' //jaxlib/tools:jaxlib_wheel\n"
           f"[EXECUTING] {bazel} build --config=y //jaxlib/tools:jax_cuda_plugin_wheel\n")
    errors = tmp_path / "errors.log"
    assert patch_stack.read_by(log, errors) == (str(bazel), [
        "@@//jaxlib:kernel.cc (null)", "@@//jaxlib:kernel.cc (null)", "@@//jaxlib:BUILD"])
    ran = [json.loads(line) for line in calls.read_text().splitlines()]
    assert ran[0][:3] == ["cquery", "--config=x", "--action_env=PATH=/a b"]
    assert ran[0][-1] == "kind('source file', deps(//jaxlib/tools:jaxlib_wheel))"
    assert ran[2] == ["query", "--repo_env=HERMETIC_CUDA_VERSION=12.9.1", "--color=no", "--consistent_labels",
                      "--output=label",
                      "buildfiles(deps(//jaxlib/tools:jaxlib_wheel + //jaxlib/tools:jax_cuda_plugin_wheel))"]
    assert "announced" in errors.read_text()
    with pytest.raises(subprocess.CalledProcessError):
        patch_stack.read_by(f"[EXECUTING] {bazel} build --fail //t:w\n", errors)
    with pytest.raises(ValueError, match="no wheel build"):
        patch_stack.read_by("INFO nothing built\n", errors)


def test_a_record_takes_the_drivers_the_checkout_tracks_and_a_vendored_file_as_its_vendors(tmp_path):
    """A cell's record (`recorded`) takes build/tools' files the checkout tracks, not what running them left (their
    __pycache__), and a file of a repository another vendors (TSL within XLA's tree, linked as its own repository)
    as the vendoring repository's, whose patches change it (row 5670's review)."""
    workspace = tmp_path / "jax"
    (workspace / "build" / "tools").mkdir(parents=True)
    (workspace / "build" / "tools" / "utils.py").write_text("tool\n")
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True)
    (workspace / "build" / "tools" / "__pycache__").mkdir()
    (workspace / "build" / "tools" / "__pycache__" / "utils.cpython-312.pyc").write_bytes(b"compiled")
    external = tmp_path / "output_base" / "external"
    (external / "xla+" / "third_party" / "tsl" / "tsl").mkdir(parents=True)
    (external / "xla+" / "third_party" / "tsl" / "tsl" / "x.h").write_text("int x;\n")
    (external / "tsl").symlink_to(external / "xla+" / "third_party" / "tsl")
    cell = patch_stack.recorded(workspace, tmp_path / "output_base", ["@@tsl//tsl:x.h"])
    assert set(cell["main"]) == {"build/tools/utils.py"}
    assert cell["external"] == {"xla+": {"third_party/tsl/tsl/x.h": patch_stack.digest(b"int x;\n")}}
