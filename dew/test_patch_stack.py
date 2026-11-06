"""The patch stack's choice of the release to carry next, over remotes' refs as git ls-remote lists them, and
the test files it runs, over a repository of its own."""

import json
import subprocess
from pathlib import Path

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
# gh as release.sh calls it: a dispatch recorded with its token, and the run named for it listed (gh having
# applied the caller's --jq) from the second listing after it, unless the fork says its runs never appear.
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
    if dispatched and listings >= 2 and dispatched[-1][-1].removeprefix("token=") in query:
        print(7)
"""


def isolated(tmp_path) -> dict:
    """An environment whose git reads no configuration but its own and installs no hooks or templates, and
    finds gh and sleep in tmp_path/bin first."""
    (tmp_path / "templates").mkdir()
    return {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1",
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
    return origin, checkout, commits, run


def test_a_release_is_pushed_and_published_only_once_every_step_before_it_has_held(tmp_path):
    """release.sh pushes the release branch and the stack branch together, the release grown on top of the one
    before so both move only forward, asks patch-stack.yml to publish and watches the run its dispatch's token
    names, only once every commit it carries has applied as a family's and the version reads exactly the new
    release: a cherry-pick that conflicts, a commit of no family, or a stack branch moved off the release it
    grows from, ends the run with nothing pushed and nothing asked; a dispatched run that never appears ends it
    within its bound, the branches left at the composed release."""
    environment = isolated(tmp_path)
    origin, checkout, commits, run = fork(tmp_path, environment)
    (tmp_path / "bin").mkdir()
    for name, text in (("gh", GH), ("sleep", "#!/bin/sh\n")):
        (tmp_path / "bin" / name).write_text(text)
        (tmp_path / "bin" / name).chmod(0o755)
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
    done = released("0.12.0.1.post40", commits["fix"])
    assert done.returncode == 0
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
    assert ["run", "watch", "7", "-R", "AshishKumar4/jax", "--exit-status"] in asked
    # The stack branch has moved off post39: a release grown from post39 again is refused, unasked.
    before, count = branches(), len(asked)
    assert released("0.12.0.1.post41", commits["fix"]).returncode != 0
    assert branches() == before and len(json.loads(calls.read_text())) == count
    # A run that never appears: the branches stay at the release they were pushed to, and the script ends.
    run(origin, "update-ref", "refs/heads/v0.12.0.1-cache-key-pool-fingerprints",
        run(origin, "rev-parse", "release/0.12.0.1.post39"))
    lost = released("0.12.0.1.post41", commits["fix"], RUNS_NEVER_APPEAR="1")
    assert lost.returncode != 0
    assert run(origin, "rev-parse", "release/0.12.0.1.post41") == run(origin, "rev-parse",
                                                                     "v0.12.0.1-cache-key-pool-fingerprints")
    assert not any(call[:2] == ["run", "watch"] for call in json.loads(calls.read_text())[count:])


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
    assert patch_stack.rebase("0.12.0", "0.12.1") == 0
    subjects = run("log", "--format=%s", "jax-v0.12.1..v0.12.1-cache-key-pool-fingerprints").splitlines()
    assert subjects == ["Release 0.12.1.post1: the stack carried from 0.12.0 onto jax-v0.12.1, a commit a family",
                        "[stack] fix-a: a is fixed"]
    assert run("show", "v0.12.1-cache-key-pool-fingerprints:jax/a.py") == "a fixed again\n"
    assert '_release_version: str | None = "0.12.1.post1"' in run(
        "show", "v0.12.1-cache-key-pool-fingerprints:jax/version.py")
