# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The tests' side of the Mosaic GPU lowering cache tests.

A test lowers a kernel in a process of its own (`child`), against a temporary
cache directory, as the cache is meant to be used: by one process after another.
A sealed installation is one whose files match their RECORD, so the cases that
need one build a site directory of their own: a copy of the jax under test, with
a RECORD made for it. `LoweringCacheTestCase` holds that site and runs the
children; the helpers below read and damage the cache's entries.
"""

import base64
import hashlib
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any

from absl.testing import absltest

SUPPORT = pathlib.Path(__file__).resolve().parent
TESTS_PALLAS = SUPPORT.parent
ROOT = SUPPORT.parents[2]
VERSION = "0.12.0.1.post51"
RESULT = "RESULT "
SM90 = "9.0"
SM80 = "8.0"

# The modules added to the sealed site's jax, as the names they have there.
_SITE_MODULES = {
    "sealed_classes.py": "jax/_src/lowering_cache_test_classes.py",
    "sealed_rules.py": "jax/_src/lowering_cache_test_rules.py",
}


def _b64(digest: bytes) -> str:
  return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# A package installed beside jax that JAX does not require (a user's, as a wheel
# installs it): its module and its dist-info, whose RECORD lists it.
_USER_PACKAGE = "user_layouts"
_USER_SOURCE = '''"""A user's layout, in a package installed as a wheel installs it."""
import dataclasses

from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax.experimental.pallas import mosaic_gpu as plgpu


@dataclasses.dataclass(frozen=True)
class InstalledLayout(gpu_core.SomeLayout):
  tag: int = 0

  def to_mgpu(self, *args, **kwargs):
    return plgpu.Layout.WGMMA.to_mgpu()
'''


def _install_user_package(dest: pathlib.Path) -> None:
  source = _USER_SOURCE.encode()
  (dest / f"{_USER_PACKAGE}.py").write_bytes(source)
  info = dest / f"{_USER_PACKAGE}-1.0.dist-info"
  info.mkdir()
  metadata = f"Metadata-Version: 2.1\nName: {_USER_PACKAGE}\nVersion: 1.0\n".encode()
  (info / "METADATA").write_bytes(metadata)
  (info / "top_level.txt").write_text(f"{_USER_PACKAGE}\n")
  rows = [
      f"{_USER_PACKAGE}.py,sha256={_b64(hashlib.sha256(source).digest())},{len(source)}",
      f"{info.name}/METADATA,sha256={_b64(hashlib.sha256(metadata).digest())},{len(metadata)}",
      f"{info.name}/top_level.txt,,",
      f"{info.name}/RECORD,,",
  ]
  (info / "RECORD").write_text("\n".join(rows) + "\n")


def make_site(dest: pathlib.Path, *, with_record: bool = True) -> None:
  """A site directory holding the jax under test, with a RECORD made for it, and a
  user's package installed beside it (`user_layouts`)."""
  shutil.copytree(
      ROOT / "jax", dest / "jax",
      ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
  for source, name in _SITE_MODULES.items():
    shutil.copyfile(SUPPORT / source, dest / name)
  info = f"jax-{VERSION}.dist-info"
  (dest / info).mkdir()
  metadata = f"Metadata-Version: 2.1\nName: jax\nVersion: {VERSION}\n".encode()
  (dest / info / "METADATA").write_bytes(metadata)
  if not with_record:
    return
  rows = []
  for path in sorted((dest / "jax").rglob("*")):
    if path.is_file():
      data = path.read_bytes()
      rows.append(
          f"{path.relative_to(dest)},sha256={_b64(hashlib.sha256(data).digest())},{len(data)}")
  rows.append(
      f"{info}/METADATA,sha256={_b64(hashlib.sha256(metadata).digest())},{len(metadata)}")
  rows.append(f"{info}/RECORD,,")
  (dest / info / "RECORD").write_text("\n".join(rows) + "\n")
  _install_user_package(dest)


def split_entry(data: bytes) -> tuple[int, dict[str, Any], bytes]:
  """An entry's (version, header, module): its magic line, a version and header
  size, a JSON header, then the module's bytecode."""
  start = data.index(b"\n") + 1
  version, size = struct.unpack("<II", data[start:start + 8])
  header = json.loads(data[start + 8:start + 8 + size])
  return version, header, data[start + 8 + size:]


def join_entry(magic: bytes, version: int, header: dict[str, Any],
               module: bytes) -> bytes:
  blob = json.dumps(header, sort_keys=True).encode()
  return magic + struct.pack("<II", version, len(blob)) + blob + module


def payload_digest(header: dict[str, Any], module: bytes) -> str:
  """The sha256 an entry holds of its whole payload: the header's fields (but
  the digest), as canonical JSON, a newline, and the module."""
  fields = {k: v for k, v in header.items() if k != "sha256"}
  canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(canonical.encode() + b"\n" + module).hexdigest()


def rewrite_entry(path: pathlib.Path, *, header: dict[str, Any] | None = None,
                  module: bytes | None = None, digest: bool = True) -> None:
  """Writes an entry again with its header's fields (and its module) changed,
  and, if `digest`, the digest of what it holds now: a damaged entry that is
  whole as far as its checksum goes."""
  data = path.read_bytes()
  magic = data[:data.index(b"\n") + 1]
  version, current, current_module = split_entry(data)
  new_header = {**current, **(header or {})}
  new_module = current_module if module is None else module
  new_header["module_size"] = len(new_module)
  if digest:
    new_header["sha256"] = payload_digest(new_header, new_module)
  path.write_bytes(join_entry(magic, version, new_header, new_module))


class LoweringCacheTestCase(absltest.TestCase):
  """A test that lowers kernels in processes of its own."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._site_root = pathlib.Path(tempfile.mkdtemp(prefix="lowering-cache-site-"))
    cls._site = cls._site_root / "site"
    cls._site.mkdir()
    make_site(cls._site)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls._site_root, ignore_errors=True)
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="lowering-cache-"))
    self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
    self.cache_dir = self.tmp / "cache"
    (self.tmp / "work").mkdir()

  # -- running a lowering in a process of its own

  def spawn_child(self, *, site: Any = "sealed", cache: bool = True,
                  env: dict[str, str] | None = None, **case: Any) -> subprocess.Popen:
    """Starts a case in a fresh process.

    `site` is "sealed" (the test's own site directory, whose RECORD is made for
    it), "overlay" (the jax under test, which no RECORD lists), or a path.
    """
    sealed = site == "sealed"
    site_dir = self._site if sealed else (None if site == "overlay" else site)
    if site_dir is not None and not isinstance(site_dir, pathlib.Path):
      site_dir = pathlib.Path(site_dir)
    spec = {
        "capability": SM90, "size": 256, "power": 2, "scale": 2.0, "pad": 0,
        "workdir": str(self.tmp / "work"),
        "site": str(site_dir) if site_dir is not None else None,
    }
    spec.update(case)
    child_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("JAX_PALLAS_LOWERING_CACHE", "MOSAIC_GPU_DUMP"))
    }
    child_env.update(
        PYTHONPATH=os.pathsep.join(
            [str(site_dir if site_dir is not None else ROOT), str(TESTS_PALLAS)]),
        JAX_PLATFORMS="cpu",
        LOWERING_CACHE_CASE=json.dumps(spec),
    )
    if cache:
      child_env["JAX_PALLAS_LOWERING_CACHE_DIR"] = str(self.cache_dir)
    child_env.update(env or {})
    return subprocess.Popen(
        [sys.executable, "-m", "mosaic_gpu_lowering_cache_support.child"],
        env=child_env, cwd=self.tmp, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)

  def collect_child(self, process: subprocess.Popen) -> dict[str, Any]:
    """What a started case reported, with its exit status and what it wrote to
    stderr (`_returncode`, `_stderr`)."""
    stdout, stderr = process.communicate()
    lines = [l for l in stdout.splitlines() if l.startswith(RESULT)]
    self.assertTrue(lines, f"no result from the process: {stderr[-3000:]}")
    result = json.loads(lines[-1][len(RESULT):])
    self.assertNotIn("crash", result, stderr[-3000:])
    result["_returncode"] = process.returncode
    result["_stderr"] = stderr[-3000:]
    return result

  def run_child(self, **kwargs: Any) -> dict[str, Any]:
    """Runs a case in a fresh process and returns what it reported."""
    return self.collect_child(self.spawn_child(**kwargs))

  def lower(self, **kwargs) -> dict[str, Any]:
    """A lowering in a process of its own: what the cache counted, and the rest."""
    run, = self.run_child(**kwargs)["runs"]
    return run

  def entries(self) -> list[pathlib.Path]:
    return sorted((self.cache_dir / "mosaic_gpu_lowering").glob("*-cache"))

  def assert_lowers_as_fresh(self, cached, fresh):
    """A lowering through the cache is the one without it: its module as text
    (the cache's modules carry no locations), or its error."""
    if "error" in fresh:
      self.assertEqual(
          (cached.get("error"), cached.get("message")),
          (fresh["error"], fresh["message"]))
    else:
      self.assertNotIn("error", cached, cached)
      self.assertEqual(
          cached["normalized"]["asm"], fresh["normalized"]["asm"])
