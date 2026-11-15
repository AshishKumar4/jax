"""dew/index.sh over a stand-in of R2's API (`aws`, a directory per bucket): which projects its root page lists, and
the order its pages are published in."""

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

from test_patch_stack import wheel

INDEX = Path(__file__).with_name("index.sh")

AWS = r"""#!/usr/bin/env python3
# aws s3api head-object|put-object|delete-object|list-objects-v2 --bucket B ..., aws s3 cp s3://B/K DEST: a
# directory per bucket, each key a file named by the key quoted whole (R2's keys `simple/` and `simple/index.html` are two objects); every
# put and delete named in order in $STORE.puts; a missing key answers as R2 does, (404).
import os, shutil, sys
from pathlib import Path
from urllib.parse import quote, unquote
store, args = Path(os.environ["STORE"]), sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]
def path(bucket, key):
    return store / bucket / quote(key, safe="")
def missing(key):
    print(f"An error occurred (404) when calling the operation: Not Found ({key})", file=sys.stderr)
    sys.exit(254)
if args[:2] == ["s3api", "head-object"]:
    key = option("--key")
    sys.exit(0) if path(option("--bucket"), key).is_file() else missing(key)
if args[:2] == ["s3api", "put-object"]:
    target = path(option("--bucket"), option("--key"))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(option("--body"), target)
    with open(f"{store}.puts", "a") as puts:
        puts.write(option("--key") + "\n")
    sys.exit(0)
if args[:2] == ["s3api", "delete-object"]:
    path(option("--bucket"), option("--key")).unlink(missing_ok=True)
    with open(f"{store}.puts", "a") as puts:
        puts.write("delete " + option("--key") + "\\n")
    sys.exit(0)
if args[:2] == ["s3api", "list-objects-v2"]:
    prefix, keys = option("--prefix"), [unquote(found.name) for found in (store / option("--bucket")).glob("*")]
    print("\t".join(sorted({prefix + key[len(prefix):].split("/")[0] + "/" for key in keys
                            if key.startswith(prefix) and "/" in key[len(prefix):]})))
    sys.exit(0)
if args[:2] == ["s3", "cp"]:
    source = next(arg for arg in args if arg.startswith("s3://"))
    bucket, _, key = source[len("s3://"):].partition("/")
    if not path(bucket, key).is_file():
        missing(key)
    shutil.copyfile(path(bucket, key), args[args.index(source) + 1])
    sys.exit(0)
sys.exit(f"aws stand-in: no {args}")
"""


def ran(tmp_path: Path, script: str, *arguments: str, new_project: str = "") -> subprocess.CompletedProcess:
    """dew/SCRIPT run over the stand-in's storage, its site a scratch directory of its own."""
    tools = tmp_path / "bin"
    tools.mkdir(exist_ok=True)
    (tools / "aws").write_text(AWS)
    (tools / "aws").chmod(0o755)
    site = tmp_path / f"site-{len(list(tmp_path.glob('site-*')))}"
    return subprocess.run(["bash", str(INDEX.with_name(script)), str(site), *arguments], capture_output=True,
                          text=True, env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}",
                                          "STORE": str(tmp_path / "r2"), "R2": "https://r2.invalid",
                                          "BUCKET": "index", "NEW_PROJECT": new_project, "TMPDIR": str(tmp_path)})


def published(tmp_path: Path, *wheels: Path, new_project: str = "") -> subprocess.CompletedProcess:
    return ran(tmp_path, "index.sh", *map(str, wheels), new_project=new_project)


def stored(tmp_path: Path, key: str) -> Path:
    return tmp_path / "r2" / "index" / quote(key, safe="")


def root(tmp_path: Path) -> str:
    return stored(tmp_path, "simple/index.html").read_text()


def test_the_root_lists_the_projects_it_listed_and_never_one_whose_publish_stopped_before_its_pages(tmp_path):
    """A project is listed on the root once its pages are published: objects of one whose first publish failed
    part-way through its uploads (packages/<name>/ without its pages) are never listed by the next publish of
    another, which keeps every project the root listed."""
    (tmp_path / "wheels").mkdir()
    first = published(tmp_path, wheel(tmp_path / "wheels", "qwix", "0.1.8"), new_project="qwix")
    assert first.returncode == 0, first.stdout + first.stderr
    stored(tmp_path, "packages/jaxlib/jaxlib-0.12.0.1-py3-none-any.whl").write_bytes(b"uploaded before the run stopped")
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
    manifest = json.loads(stored(tmp_path, "packages/jaxlib/manifest.json").read_text())
    assert list(manifest) == ["jaxlib-0.12.0.1-py3-none-any.whl"], manifest


def test_an_unlisted_project_leaves_the_root_first_then_its_pages_and_manifest_and_is_new_again(tmp_path):
    """The undoing of a first publication (unlist.sh): the root rewritten without the project before its pages
    go, so no root lists a project without pages; its pages under both names deleted, so an installer listing the
    index first resolves it from PyPI again; its manifest deleted, so publishing it again needs NEW_PROJECT; every
    other project still listed, the project's files kept."""
    (tmp_path / "wheels").mkdir()
    for name in ("qwix", "jaxlib"):
        first = published(tmp_path, wheel(tmp_path / "wheels", name, "0.12.0.1"), new_project=name)
        assert first.returncode == 0, first.stdout + first.stderr
    (tmp_path / "r2.puts").unlink()
    undone = ran(tmp_path, "unlist.sh", "jaxlib")
    assert undone.returncode == 0, undone.stdout + undone.stderr
    listed = root(tmp_path)
    assert 'href="qwix/"' in listed and "jaxlib" not in listed, listed
    writes = (tmp_path / "r2.puts").read_text().splitlines()
    assert writes == ["simple/", "simple/index.html", "delete simple/jaxlib/", "delete simple/jaxlib/index.html",
                      "delete packages/jaxlib/manifest.json"], writes
    assert stored(tmp_path, "packages/jaxlib/jaxlib-0.12.0.1-py3-none-any.whl").is_file()
    (tmp_path / "r2.puts").unlink()
    again = published(tmp_path, wheel(tmp_path / "wheels", "jaxlib", "0.12.0.1"))
    assert again.returncode != 0 and not (tmp_path / "r2.puts").exists(), again.stdout + again.stderr
