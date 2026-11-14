"""dew/debug_split.py: a wheel's shared objects stripped in place, their debug information beside them by build-id."""

import base64
import hashlib
import subprocess
import zipfile
from pathlib import Path

import debug_split


def test_a_wheels_object_ships_stripped_and_its_line_tables_are_read_from_its_build_id_file(tmp_path: Path):
    """An object built with line tables, in a wheel, is stripped of them and of its symbol table there, its RECORD
    holding its new hash; its debug file, under .build-id named by its build-id, holds the line tables and the
    function's name, which the stripped object's link names."""
    source = tmp_path / "kernel.c"
    source.write_text("int dew_kernel(int x) { return x * 3 + 1; }\n")
    built = tmp_path / "libkernel.so"
    subprocess.run(["cc", "-g1", "-O1", "-shared", "-fPIC", "-Wl,--build-id", str(source), "-o", str(built)],
                   check=True)
    identity = debug_split.build_id(built)
    wheel = tmp_path / "dist/jax_cuda12_plugin-0.1-cp312-cp312-manylinux_2_27_x86_64.whl"
    wheel.parent.mkdir()
    record = "jax_cuda12_plugin-0.1.dist-info/RECORD"
    with zipfile.ZipFile(wheel, "w") as made:
        made.writestr("jax_plugins/libkernel.so", built.read_bytes())
        made.writestr(record, f"jax_plugins/libkernel.so,sha256=old,1\n{record},,\n")
    [debug] = debug_split.rewrite(wheel, tmp_path / "debug")

    assert debug == tmp_path / "debug/.build-id" / identity[:2] / f"{identity[2:]}.debug"
    with zipfile.ZipFile(wheel) as shipped:
        stripped = shipped.read("jax_plugins/libkernel.so")
        lines = shipped.read(record).decode().splitlines()
    (tmp_path / "stripped.so").write_bytes(stripped)

    def sections(path: Path) -> str:
        return subprocess.run(["readelf", "-S", "-W", str(path)], capture_output=True, text=True, check=True).stdout

    assert ".debug_line" not in sections(tmp_path / "stripped.so") and ".symtab" not in sections(
        tmp_path / "stripped.so")
    assert ".debug_line" in sections(debug) and debug_split.build_id(tmp_path / "stripped.so") == identity
    digest = base64.urlsafe_b64encode(hashlib.sha256(stripped).digest()).rstrip(b"=").decode()
    assert lines == [f"jax_plugins/libkernel.so,sha256={digest},{len(stripped)}", f"{record},,"]
    # The link section holds the debug file's name and its CRC's bytes, no text of their own.
    link = subprocess.run(["readelf", "-p", ".gnu_debuglink", str(tmp_path / "stripped.so")], capture_output=True,
                          check=True).stdout.decode(errors="replace")
    assert f"{identity[2:]}.debug" in link
    symbols = subprocess.run(["readelf", "-s", "-W", str(debug)], capture_output=True, text=True, check=True).stdout
    assert "dew_kernel" in symbols
