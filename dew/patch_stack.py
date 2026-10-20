"""Dew's patch stack on JAX: the fixes Dew carries in this fork until a JAX release has them, moved from one
release to the next.

    python dew/patch_stack.py next              # "OLD NEW": the newest carried release and a newer one
    python dew/patch_stack.py rebase OLD NEW    # in a clone holding jax-vOLD, jax-vNEW and the OLD branch
    python dew/patch_stack.py index MANIFEST SITE FILE...   # the package index, FILEs added to MANIFEST
    python dew/patch_stack.py index MANIFEST SITE --yank FILE... --reason WHY   # FILEs yanked (PEP 592)

`next` reads the releases upstream has tagged (jax-vX.Y.Z) and the stacks this fork carries
(vX.Y.Z-cache-key-pool-fingerprints) and prints the newest stack's release and the newest release, where
that release has no stack yet; nothing where it has.

`rebase` takes the commits of vOLD-cache-key-pool-fingerprints past the tag jax-vOLD and applies each, in
order, onto jax-vNEW as vNEW-cache-key-pool-fingerprints:

- a commit whose change jax-vNEW already holds is dropped: an equal patch (`git cherry`), or one whose
  fix applies as nothing;
- its fix is applied as a three-way merge, and where it sets the build's version (jax/version.py's
  `_release_version`), version.py is jax-vNEW's with the commit's lines in place of that line, the
  version NEW.postN, N counting the version-setting commits kept, so each build sorts after the
  release it patches and satisfies `jax>=NEW`;
- OLD in each message, and in version.py's comment, becomes NEW; each keeps its author and dates, and
  names the commit it was taken from.

It prints one line a commit (`kept`, `dropped` and why) and exits 0 with the branch made; 2 where a commit
does not apply, after printing its conflict and leaving no branch; 3 where every fix was dropped, so only
version commits would remain and the fork has nothing left to carry for NEW.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

VERSION = Path("jax/version.py")
RELEASE = re.compile(r'^_release_version: str \| None = "[^"]*"$', re.MULTILINE)
UPSTREAM = "https://github.com/jax-ml/jax.git"
FORK = "https://github.com/AshishKumar4/jax.git"
TAG = re.compile(r"refs/tags/jax-v(\d+)\.(\d+)\.(\d+)$")
STACK = re.compile(r"refs/heads/v(\d+)\.(\d+)\.(\d+)-cache-key-pool-fingerprints$")


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True).stdout


def branch(version: str) -> str:
    return f"v{version}-cache-key-pool-fingerprints"


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


def rebase(old: str, new: str) -> int:
    commits = git("rev-list", "--reverse", f"jax-v{old}..{branch(old)}").split()
    held = upstreamed(old, new)
    target = branch(new)
    git("checkout", "-q", "--detach", f"jax-v{new}")
    posts, fixes = 0, 0
    for commit in commits:
        title = git("log", "-1", "--format=%s", commit).strip()
        if commit in held:
            print(f"dropped {commit[:9]} (jax-v{new} holds the same patch): {title}")
            continue
        paths = git("diff", "--name-only", f"{commit}^", commit).split()
        fix = [path for path in paths if path != str(VERSION)]
        if fix:
            # The fix applied as a three-way merge; the version is written apart, each commit's own.
            patch = git("diff", "--binary", f"{commit}^", commit, "--", *fix)
            applied = subprocess.run(["git", "apply", "--3way", "--index"], input=patch, capture_output=True,
                                     text=True)
            if applied.returncode != 0:
                print(f"conflict {commit[:9]}: {title}\n{applied.stderr}{git('diff', check=False)}")
                git("reset", "-q", "--hard")
                return 2
            if not git("diff", "--cached", "--name-only").strip():
                print(f"dropped {commit[:9]} (applies as nothing on jax-v{new}): {title}")
                continue
            fixes += 1
        message = git("log", "-1", "--format=%B", commit).replace(old, new)
        if str(VERSION) in paths:
            posts += 1
            VERSION.write_text(versioned(old, new, commit, posts))
            git("add", str(VERSION))
            message = re.sub(rf"{re.escape(new)}\.post\d+", f"{new}.post{posts}", message)
        author = git("log", "-1", "--format=%an <%ae>", commit).strip()
        date = git("log", "-1", "--format=%aD", commit).strip()
        with tempfile.TemporaryDirectory() as scratch:
            written = Path(scratch) / "message"
            written.write_text(f"{message.rstrip()}\n\n(cherry picked from commit {commit})\n")
            git("commit", "-q", "--no-verify", "-F", str(written), f"--author={author}", f"--date={date}")
        print(f"kept {commit[:9]} as {git('rev-parse', '--short=9', 'HEAD').strip()}: {title}")
    if fixes == 0:
        print(f"jax-v{new} holds every fix: nothing left to carry")
        git("checkout", "-q", "--detach", f"jax-v{new}")
        return 3
    git("checkout", "-q", "-B", target)
    print(f"{target} at {git('rev-parse', 'HEAD').strip()}, version {new}.post{posts}")
    return 0


def versions(remote: str, pattern: re.Pattern) -> list[tuple[int, int, int]]:
    """The X.Y.Z of each ref of `remote` that `pattern` matches, sorted."""
    listed = git("ls-remote", remote).split()
    return sorted(tuple(int(part) for part in found.groups())  # type: ignore[misc]
                  for found in map(pattern.match, listed[1::2]) if found)


def following() -> str:
    """"OLD NEW" where the newest release NEW has no stack and OLD is the newest stack's release, else ""."""
    releases, stacks = versions(UPSTREAM, TAG), versions(FORK, STACK)
    newest = releases[-1]
    if not stacks or newest in stacks or newest < stacks[-1]:
        return ""
    return " ".join(".".join(map(str, version)) for version in (stacks[-1], newest))


def metadata(wheel: Path) -> bytes:
    """A wheel's core metadata, its .dist-info/METADATA (PEP 658)."""
    with zipfile.ZipFile(wheel) as archive:
        (name,) = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        return archive.read(name)


def index(manifest: Path, site: Path, files: list[Path], yank: list[str] = (), reason: str = "") -> None:
    """The package index under `site`, PEP 503's simple API for jax alone: simple/index.html and
    simple/jax/index.html, each file linked with its digest, its Requires-Python and its metadata's digest
    (PEP 658, the metadata beside it as FILE.metadata). `manifest` keeps every file the index has listed, so
    a run adds its own to them; a file it lists already is never replaced, as an installer may hold its
    digest. Each new FILE is copied, with its metadata, under site/packages/jax and named on stdout.

    Each file named in `yank`, one the manifest lists, is marked yanked for `reason` (PEP 592's
    data-yanked): an installer resolving a range passes it over, and one pinned to exactly it still gets it."""
    listed = json.loads(manifest.read_text()) if manifest.is_file() else {}
    unlisted = sorted(set(yank) - set(listed))
    if unlisted:
        raise SystemExit(f"cannot yank {', '.join(unlisted)}: the index does not list it")
    for name in yank:
        listed[name]["yanked"] = reason
    packages = site / "packages" / "jax"
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
        f'<a href="../../packages/jax/{name}#sha256={entry["sha256"]}"'
        f' data-requires-python="{html.escape(entry["requires_python"])}"'
        f' data-dist-info-metadata="sha256={entry["metadata_sha256"]}"'
        f' data-core-metadata="sha256={entry["metadata_sha256"]}"'
        + (f' data-yanked="{html.escape(entry["yanked"])}"' if "yanked" in entry else "")
        + f'>{name}</a><br>\n'
        for name, entry in sorted(listed.items()))
    page = ('<!DOCTYPE html>\n<html><head><meta name="pypi:repository-version" content="1.1">'
            "<title>{title}</title></head>\n<body><h1>{title}</h1>\n{body}</body></html>\n")
    (site / "simple" / "jax").mkdir(parents=True, exist_ok=True)
    (site / "simple" / "index.html").write_text(page.format(title="Simple index", body='<a href="jax/">jax</a><br>\n'))
    (site / "simple" / "jax" / "index.html").write_text(page.format(title="Links for jax", body=links))


def main() -> int:
    parser = argparse.ArgumentParser(prog="patch_stack.py")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("next")
    rebasing = commands.add_parser("rebase")
    rebasing.add_argument("old")
    rebasing.add_argument("new")
    indexing = commands.add_parser("index")
    indexing.add_argument("manifest", type=Path)
    indexing.add_argument("site", type=Path)
    indexing.add_argument("files", type=Path, nargs="*")
    indexing.add_argument("--yank", nargs="+", default=[], metavar="FILE", help="the listed files to mark yanked")
    indexing.add_argument("--reason", default="", help="why they are yanked, shown by installers")
    args = parser.parse_args()
    if args.command == "next":
        print(following())
        return 0
    if args.command == "index":
        index(args.manifest, args.site, args.files, args.yank, args.reason)
        return 0
    return rebase(args.old, args.new)


if __name__ == "__main__":
    sys.exit(main())
