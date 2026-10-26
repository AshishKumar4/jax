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
