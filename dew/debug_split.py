"""Each shared object of the wheels in a directory split into a stripped object, kept in its wheel, and its debug
information, a file of its own named by its GNU build-id (`.build-id/xx/yyyy.debug`, as gdb, eu-stack and pystack
look one up), so a release's wheels install as small as stock while a stack captured on a host running them can be
read (row 6013: the plugin's native frames were unreadable). The plugins are built with line tables (-g1,
dew/build_plugin.sh) and every split object keeps a .gnu_debuglink to its debug file's name.

    python3 dew/debug_split.py DIST DEBUG     # every DIST/*.whl's *.so split; debug files under DEBUG/.build-id/
"""

from __future__ import annotations

import base64
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

BUILD_ID = re.compile(r"Build ID: ([0-9a-f]+)")
"""readelf -n's line naming an object's GNU build-id."""


def build_id(path: Path) -> str:
    """The GNU build-id of ELF object `path`, as readelf reads its note."""
    notes = subprocess.run(["readelf", "-n", str(path)], capture_output=True, text=True, check=True).stdout
    found = BUILD_ID.search(notes)
    if found is None:
        raise SystemExit(f"debug_split: {path.name} has no GNU build-id (link with --build-id)")
    return found[1]


def split(path: Path, debug: Path) -> Path:
    """Object `path`'s debug information moved to its build-id's file under `debug`, the object stripped of it and
    of its symbol table (as stock wheels ship), a link to the debug file kept: the debug file's path."""
    found = build_id(path)
    target = debug / ".build-id" / found[:2] / f"{found[2:]}.debug"
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["objcopy", "--only-keep-debug", "--compress-debug-sections", str(path), str(target)], check=True)
    subprocess.run(["objcopy", "--strip-all", f"--add-gnu-debuglink={target}", str(path)], check=True)
    return target


def record_line(name: str, data: bytes) -> str:
    """A wheel RECORD line for member `name` holding `data` (PEP 376/427: urlsafe base64 sha256 without padding)."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def rewrite(wheel: Path, debug: Path) -> list[Path]:
    """Wheel `wheel` with each of its *.so members split (`split`), its RECORD holding their new hashes: the debug
    files written."""
    written: list[Path] = []
    with tempfile.TemporaryDirectory() as scratch, zipfile.ZipFile(wheel) as source:
        members = source.infolist()
        contents = {member.filename: source.read(member) for member in members}
        for name in [name for name in contents if name.endswith(".so") or ".so." in Path(name).name]:
            local = Path(scratch) / Path(name).name
            local.write_bytes(contents[name])
            written.append(split(local, debug))
            contents[name] = local.read_bytes()
        [record] = [name for name in contents if name.endswith(".dist-info/RECORD")]
        lines = [line for line in contents[record].decode().splitlines() if line]
        contents[record] = ("\n".join(
            record_line(line.split(",")[0], contents[line.split(",")[0]])
            if line.split(",")[0] in contents and line.split(",")[0] != record else line
            for line in lines) + "\n").encode()
        part = wheel.with_suffix(".part")
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as target:
            for member in members:
                target.writestr(member, contents[member.filename])
    shutil.move(part, wheel)
    return written


if __name__ == "__main__":
    dist, debug = Path(sys.argv[1]), Path(sys.argv[2])
    for wheel in sorted(dist.glob("*.whl")):
        for path in rewrite(wheel, debug):
            print(f"{wheel.name}: {path.relative_to(debug)}")
