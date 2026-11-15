"""dew/index.sh over a stand-in of R2's API (`aws`, a directory per bucket): which projects its root page lists, and
the order its pages are published in."""

import json
import os
import subprocess
from pathlib import Path

from test_patch_stack import wheel

INDEX = Path(__file__).with_name("index.sh")

AWS = r"""#!/usr/bin/env python3
# aws s3api head-object|put-object --bucket B --key K [--body F] ..., aws s3 cp s3://B/K DEST: a directory per
# bucket ($STORE/B/K); every put named in order in $STORE.puts; a missing key answers as R2 does, (404).
import os, shutil, sys
from pathlib import Path
store, args = Path(os.environ["STORE"]), sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]
def missing(key):
    print(f"An error occurred (404) when calling the operation: Not Found ({key})", file=sys.stderr)
    sys.exit(254)
if args[:2] == ["s3api", "head-object"]:
    key = option("--key")
    sys.exit(0) if (store / option("--bucket") / key).is_file() else missing(key)
if args[:2] == ["s3api", "put-object"]:
    target = store / option("--bucket") / option("--key")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(option("--body"), target)
    with open(f"{store}.puts", "a") as puts:
        puts.write(option("--key") + "\n")
    sys.exit(0)
if args[:2] == ["s3", "cp"]:
    source = next(arg for arg in args if arg.startswith("s3://"))[len("s3://"):]
    destination = args[args.index(f"s3://{source}") + 1]
    found = store / source
    if not found.is_file():
        missing(source)
    shutil.copyfile(found, destination)
    sys.exit(0)
sys.exit(f"aws stand-in: no {args}")
"""


def published(tmp_path: Path, *wheels: Path, new_project: str = "") -> subprocess.CompletedProcess:
    tools = tmp_path / "bin"
    tools.mkdir(exist_ok=True)
    (tools / "aws").write_text(AWS)
    (tools / "aws").chmod(0o755)
    site = tmp_path / "site"
    return subprocess.run(["bash", str(INDEX), str(site), *map(str, wheels)], capture_output=True, text=True,
                          env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "STORE": str(tmp_path / "r2"),
                               "R2": "https://r2.invalid", "BUCKET": "index", "NEW_PROJECT": new_project,
                               "TMPDIR": str(tmp_path)})


def root(tmp_path: Path) -> str:
    return (tmp_path / "r2" / "index" / "simple" / "index.html").read_text()


def test_the_root_lists_the_projects_it_listed_and_never_one_whose_publish_stopped_before_its_pages(tmp_path):
    """A project is listed on the root once its pages are published: objects of one whose first publish failed
    part-way through its uploads (packages/<name>/ without its pages) are never listed by the next publish of
    another, which keeps every project the root listed."""
    (tmp_path / "wheels").mkdir()
    first = published(tmp_path, wheel(tmp_path / "wheels", "qwix", "0.1.8"), new_project="qwix")
    assert first.returncode == 0, first.stdout + first.stderr
    stopped = tmp_path / "r2" / "index" / "packages" / "jaxlib" / "jaxlib-0.12.0.1-py3-none-any.whl"
    stopped.parent.mkdir(parents=True)
    stopped.write_bytes(b"uploaded before the run stopped")
    second = published(tmp_path, wheel(tmp_path / "wheels", "jax", "0.12.0.1"), new_project="jax")
    assert second.returncode == 0, second.stdout + second.stderr
    listed = root(tmp_path)
    assert 'href="qwix/"' in listed and 'href="jax/"' in listed and "jaxlib" not in listed, listed


def test_a_projects_pages_are_published_before_the_root_that_lists_it(tmp_path):
    """Every file and the manifest first, then the project's page under both its names, then the root's: a root
    that lists a project has its page already, wherever a run stops."""
    (tmp_path / "wheels").mkdir()
    ran = published(tmp_path, wheel(tmp_path / "wheels", "jaxlib", "0.12.0.1"), new_project="jaxlib")
    assert ran.returncode == 0, ran.stdout + ran.stderr
    puts = (tmp_path / "r2.puts").read_text().split()
    pages = [key for key in puts if key.startswith("simple")]
    assert pages == ["simple/jaxlib/", "simple/jaxlib/index.html", "simple/", "simple/index.html"], puts
    assert puts.index("packages/jaxlib/manifest.json") < puts.index(pages[0])
    manifest = json.loads((tmp_path / "r2" / "index" / "packages" / "jaxlib" / "manifest.json").read_text())
    assert list(manifest) == ["jaxlib-0.12.0.1-py3-none-any.whl"], manifest
