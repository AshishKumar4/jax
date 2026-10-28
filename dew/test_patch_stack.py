"""The patch stack's choice of the release to carry next, over remotes' refs as git ls-remote lists them, and
the test files it runs, over a repository of its own."""

import subprocess

import patch_stack


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
