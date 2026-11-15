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

"""The sealed installation: the code a Mosaic GPU lowering runs is the installed one.

Each process hashes every file the ``RECORD`` manifests of the installed ``jax``,
``jaxlib`` and CUDA plugin distributions list, and compares it with the manifest
(the installed bytes, not their size and modification time). A file that differs,
a manifest that is missing (an editable or source install) or a file that cannot
be read leaves the process unsealed; the digest of a sealed one is the hash of
the manifests. ``Seal.verify_loaded`` holds a jax or jaxlib module loaded after
that to the same RECORD.

What a key names is then known by where it is defined, which the sealed code
fixes: ``Seal.class_identity`` and ``Seal.function_identity`` name a class or a
function by the module that owns it (a file of the installation, of the
standard library or of an installed package, whose version is part of every key),
and ``Seal.code_identity`` a code object of the installation by its file, its
line, its name and a hash of what it does (two lambdas of one file and one name
are two). A class or function that no such module owns has no identity, which
makes what holds it uncacheable.

A process may declare packages of its own sealed by their source
(``JAX_PALLAS_LOWERING_CACHE_SOURCES``, their top-level names separated by
commas): each declared package's every Python file is hashed once, and its
code, classes and primitives are named as the installation's are, with the
digest of the package's whole tree in their identity. An entry that holds such
code is then read by a process of the same tree alone; one that holds none is
unaffected. The declaration is the process's statement that the package's code
a lowering runs (an ``inline_mgpu`` body) reads nothing but its arguments, its
closure and the package's code.
"""

from __future__ import annotations

import base64
import concurrent.futures
import enum
import hashlib
import importlib.metadata
import importlib.util
import os
import re
import struct
import sys
import sysconfig
import threading
import time
import types
from collections.abc import Sequence
from typing import Any

# The distributions whose installed files are the code that lowers.
_SEALED_DISTRIBUTIONS = re.compile(r"^(jax|jaxlib|jax-cuda\d+-.+)$")
_REQUIRED_DISTRIBUTIONS = ("jax", "jaxlib")
# Whose modules must all be files of those distributions.
SEALED_MODULE_ROOTS = frozenset({"jax", "jaxlib"})

# Private: where the distributions' manifests are looked for. None is the
# interpreter's sys.path. A test sets it to a site directory it made.
_SEAL_ROOTS: Sequence[str] | None = None

_LEN = struct.Struct("<Q")

EXTENSION_SUFFIXES = (".so", ".pyd", ".dylib")


class Origin(enum.Enum):
  """Whose code a module is."""
  # A Python file of the sealed installation: its code is what RECORD hashes.
  SEALED = "sealed"
  # A compiled module of the sealed installation (or one inside it): bytes RECORD hashes.
  EXTENSION = "extension"
  # The interpreter's and its standard library: the Python version is in every key.
  STANDARD = "standard"
  # A file of a distribution JAX itself requires (`DEPENDENCIES`): its version is
  # in the key. Another installed package's code is a user's (UNKNOWN): the key
  # names a class of it by its name and fields alone, not by what its methods
  # read, which the lowering may call.
  THIRD_PARTY = "third party"
  # A Python file of a package the process declares sealed by its source
  # (`SOURCES_VARIABLE`): the digest of its whole tree is in the identity.
  SOURCE = "source"
  # Anything else: a file of the user's, a module made at run time.
  UNKNOWN = "unknown"


SOURCES_VARIABLE = "JAX_PALLAS_LOWERING_CACHE_SOURCES"
"""The environment variable that names the packages a process declares sealed by
their source: their top-level names, separated by commas."""

DEPENDENCIES = frozenset({"numpy", "ml-dtypes"})
"""The installed distributions, by normalized name, whose classes and functions a
key may name by their module and version: those JAX's lowering takes values of."""


class Unsealed(Exception):
  """The code this lowering runs is not the sealed installation's."""


class NotSealed(Exception):
  """A function, a class or a piece of code is not the sealed installation's."""


def _framed(*parts: bytes) -> bytes:
  return b"".join(_LEN.pack(len(p)) + p for p in parts)


class Seal:
  """The installed code's identity, or why it has none."""

  def __init__(
      self, digest: bytes | None, reason: str | None, files: dict[str, str],
      seconds: float, file_count: int, byte_count: int,
      hashes: dict[str, str] | None = None,
      verified_modules: frozenset[str] = frozenset(),
      sources: dict[str, tuple[str, str]] | None = None,
      source_digests: dict[str, bytes] | None = None,
  ):
    # The sha256 of the distributions' RECORDs; None if unsealed.
    self.digest = digest
    # Why it is unsealed.
    self.reason = reason
    # The installed files RECORD lists, as hashed, each by the name RECORD gives
    # it (relative to the directory the distribution is installed in).
    self.files = files
    # What sealing cost this process, and what it hashed.
    self.seconds = seconds
    self.file_count = file_count
    self.byte_count = byte_count
    # The sha256 RECORD lists for each file, by the name it gives it.
    self._hashes = hashes or {}
    # The loaded modules whose files were checked against RECORD, and the lock
    # that keeps two threads from checking one.
    self._verified = set(verified_modules)
    self._verify_lock = threading.Lock()
    self._relative: dict[str, str | None] = {}
    self._source_of: dict[str, tuple[str, str] | None] = {}
    # The declared source packages' files (`SOURCES_VARIABLE`), each by its path
    # to its package and its name in it, and each package's tree digest.
    self._sources = sources or {}
    self.source_digests = source_digests or {}
    # By id, with the code kept, as code objects compare by value and not by file.
    self._code_identities: dict[int, tuple[types.CodeType, bytes]] = {}
    # What a module is, by name and the module the name was found to be.
    self._origins: dict[tuple[str, int], tuple[Origin, tuple[str, str] | None]] = {}

  # -- files

  def relative(self, file: str | None) -> str | None:
    """The name RECORD gives `file`, if it is one of the sealed installation's files."""
    if not file:
      return None
    if file not in self._relative:
      self._relative[file] = (
          self.files.get(os.path.normpath(file))
          or self.files.get(os.path.realpath(file))
      )
    return self._relative[file]

  def holds(self, file: str | None) -> bool:
    return self.relative(file) is not None

  def source(self, file: str | None) -> tuple[str, str] | None:
    """The declared source package `file` is of and its name in it, if any.

    Read for every primitive a key's jaxpr binds, so it is answered once a
    file: resolving a path's links was 0.15 of a 0.48 s key."""
    if not file:
      return None
    if file not in self._source_of:
      self._source_of[file] = (
          self._sources.get(os.path.normpath(file))
          or self._sources.get(os.path.realpath(file))
      )
    return self._source_of[file]

  def source_identity(self, package: str) -> bytes:
    """A declared source package's identity: its name and its tree's digest."""
    return _framed(b"source", package.encode(), self.source_digests[package])

  def module_origin(self, name: str) -> tuple[Origin, tuple[str, str] | None]:
    """Whose code the module `name` is, and the distribution (name, version) if
    it is another one's."""
    module = sys.modules.get(name)
    if module is None:
      return Origin.UNKNOWN, None
    key = (name, id(module))
    found = self._origins.get(key)
    if found is None:
      found = self._origins[key] = self._origin(name, module)
    return found

  def _origin(self, name: str, module: types.ModuleType):
    file = getattr(module, "__file__", None)
    top = name.partition(".")[0]
    if file is None:
      if name in sys.builtin_module_names or top in sys.stdlib_module_names:
        return Origin.STANDARD, None
      # A submodule an extension module made (nanobind's `ir` of `_mlir`): the
      # origin of the module it is in. A module with no file inside a Python
      # package is one that was made at run time.
      parent = name.rpartition(".")[0]
      if parent:
        origin, dist = self.module_origin(parent)
        if origin is Origin.EXTENSION:
          return origin, dist
      return Origin.UNKNOWN, None
    if self.holds(file):
      return (Origin.EXTENSION if file.endswith(EXTENSION_SUFFIXES)
              else Origin.SEALED), None
    source = self.source(file)
    if source is not None:
      return Origin.SOURCE, (source[0], self.source_digests[source[0]].hex())
    if top in sys.stdlib_module_names and _in_standard_library(file):
      return Origin.STANDARD, None
    dist = _distribution_of(top, file)
    if dist is not None and _normalize_distribution(dist[0]) in DEPENDENCIES:
      return Origin.THIRD_PARTY, dist
    return Origin.UNKNOWN, None

  def verify_loaded(self) -> None:
    """Checks the modules loaded since the last check against RECORD.

    Sealing checks every file the installation lists and the modules loaded at
    the time. A jax or jaxlib module loaded after that (a lazy import, or one the
    process was made to find in its path) is as much code a lowering runs: it must
    be a file RECORD lists, as RECORD lists it. Raises Unsealed if it is not.
    """
    with self._verify_lock:
      for name, module in list(sys.modules.items()):
        if name in self._verified:
          continue
        file = getattr(module, "__file__", None)
        if name.partition(".")[0] not in SEALED_MODULE_ROOTS or file is None:
          self._verified.add(name)
          continue
        relative = self.relative(file)
        if relative is None:
          raise Unsealed(
              f"module {name} was loaded after sealing from {file}, which no"
              " installed RECORD lists")
        problem, _ = _hash_file((file, self._hashes[relative]))
        if problem is not None:
          raise Unsealed(f"module {name} was loaded after sealing: {problem}")
        self._verified.add(name)

  # -- code

  def _constant(self, value: Any) -> bytes:
    t = type(value)
    if value is None:
      return b"N"
    if t is bool:
      return b"B1" if value else b"B0"
    if t is int:
      return b"I" + str(value).encode()
    if t is float:
      return b"F" + struct.pack("<d", value)
    if t is complex:
      return b"C" + struct.pack("<dd", value.real, value.imag)
    if t is str:
      return b"S" + value.encode("utf-8", "surrogatepass")
    if t is bytes:
      return b"Y" + value
    if value is Ellipsis:
      return b"E"
    if t is tuple:
      return b"T" + _framed(*(self._constant(v) for v in value))
    if t is frozenset:
      return b"Z" + _framed(*sorted(self._constant(v) for v in value))
    if t is types.CodeType:
      return b"K" + self.code_identity(value)
    raise NotSealed(f"a code constant of the type {t.__qualname__} has no identity")

  def code_identity(self, code: types.CodeType) -> bytes:
    """What the code object is: where in which sealed file, and what it does.

    The file it is of (relative to its distribution), the line it starts at and
    its name say which definition of the file it is; a hash of everything it
    does (bytecode, constants, the names it reads, its variables and exception
    table, nested code included) says it is the one that file defines, and tells
    two lambdas of one line apart.
    """
    memo = self._code_identities.get(id(code))
    if memo is not None and memo[0] is code:
      return memo[1]
    relative = self.relative(code.co_filename)
    source = None if relative is not None else self.source(code.co_filename)
    if relative is None and source is None:
      raise NotSealed(
          f"{code.co_qualname} is defined in {code.co_filename}, which the"
          " sealed installation does not hold"
      )
    body = hashlib.sha256()
    body.update(_framed(
        code.co_qualname.encode(),
        str((code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
             code.co_flags)).encode(),
        code.co_code,
        self._constant(code.co_names),
        self._constant(code.co_varnames),
        self._constant(code.co_freevars),
        self._constant(code.co_cellvars),
        code.co_exceptiontable,
        self._constant(code.co_consts),
    ))
    where = (relative.encode() if source is None else
             self.source_identity(source[0]) + source[1].encode())
    identity = _framed(
        where, str(code.co_firstlineno).encode(), code.co_name.encode(),
        body.digest(),
    )
    self._code_identities[id(code)] = (code, identity)
    return identity

  def function_identity(self, fn: types.FunctionType) -> bytes:
    """The identity of a function, by the file it is defined in.

    A function of a file of the sealed installation is its code
    (`code_identity`). One of the standard library or of a distribution JAX
    requires (`DEPENDENCIES`) is named by the module that owns it (the
    distribution's version is part of every key). One of a declared source
    package is its code, with the package's tree digest. Any other (a function a
    user wrote, installed or not, code made at run time) has none.
    """
    code = fn.__code__
    if self.holds(code.co_filename) or self.source(code.co_filename):
      return self.code_identity(code)
    module = fn.__module__ or ""
    if _in_standard_library(code.co_filename) and (
        module.partition(".")[0] in sys.stdlib_module_names):
      return _framed(b"standard", module.encode(), fn.__qualname__.encode())
    dist = _distribution_of(module.partition(".")[0], code.co_filename)
    if dist is not None and _normalize_distribution(dist[0]) in DEPENDENCIES:
      return _framed(
          b"installed", dist[0].encode(), dist[1].encode(), module.encode(),
          fn.__qualname__.encode(), str(code.co_firstlineno).encode())
    raise NotSealed(
        f"{code.co_qualname} is defined in {code.co_filename}, which is no file"
        " of the sealed installation, the standard library or a distribution JAX"
        " requires"
    )

  # -- classes

  def class_identity(self, cls: type) -> bytes:
    """The identity of a class, by the module that owns it.

    A class is named by its module and its qualified name, and is the object its
    module holds under that name. Its module must be a file of the sealed
    installation (the class is then the code the seal hashes), a compiled module
    of it, the standard library or a distribution JAX requires (`DEPENDENCIES`,
    whose version is part of every key). A class a user defined (its module is
    one of theirs, installed or not, but a declared source package's, whose tree
    digest is in its identity), one made inside a function, or one that its
    module does not hold under its name has no identity.
    """
    qualname = cls.__qualname__
    module = sys.modules.get(cls.__module__)
    if module is None:
      raise NotSealed(f"the class {cls.__module__}.{qualname} is of no loaded module")
    origin, dist = self.module_origin(cls.__module__)
    if origin is Origin.UNKNOWN:
      raise NotSealed(
          f"the class {cls.__module__}.{qualname} is of a module that is no file of"
          " the sealed installation, the standard library or a distribution JAX"
          " requires"
      )
    if origin is not Origin.EXTENSION:
      # (A class of a compiled module may be called what its Python wrapper
      # replaced it with: it is named by what the module says.)
      found: Any = module
      for part in qualname.split("."):
        found = getattr(found, part, None)
      if found is not cls:
        raise NotSealed(
            f"the class {cls.__module__}.{qualname} is not the object that module"
            " holds under that name"
        )
    return _framed(
        origin.value.encode(), f"{cls.__module__}:{qualname}".encode(),
        *((dist[0].encode(), dist[1].encode()) if dist else ()))


def _in_standard_library(file: str) -> bool:
  standard = os.path.realpath(sysconfig.get_paths()["stdlib"]) + os.sep
  return os.path.realpath(file).startswith(standard)


_DISTRIBUTIONS: dict[str, list[str]] | None = None
_DISTRIBUTION_FILES: dict[str, set[str]] = {}
_DISTRIBUTION_VERSIONS: dict[str, str] = {}


def _is_file_of_distribution(dist: str, file: str) -> bool:
  """Whether the installed distribution `dist` lists `file` (or a file that is it)."""
  files = _DISTRIBUTION_FILES.get(dist)
  if files is None:
    d = importlib.metadata.distribution(dist)
    files = {
        os.path.realpath(str(d.locate_file(f))) for f in d.files or ()
    }
    _DISTRIBUTION_FILES[dist] = files
    _DISTRIBUTION_VERSIONS[dist] = d.version
  return os.path.realpath(file) in files


def _distribution_of(top: str, file: str) -> tuple[str, str] | None:
  """The installed distribution that lists `file` of the package `top`, and its version."""
  global _DISTRIBUTIONS
  if _DISTRIBUTIONS is None:
    _DISTRIBUTIONS = dict(importlib.metadata.packages_distributions())
  for dist in _DISTRIBUTIONS.get(top, ()):
    if _is_file_of_distribution(dist, file):
      return dist, _DISTRIBUTION_VERSIONS[dist]
  return None


def _normalize_distribution(name: str) -> str:
  return re.sub(r"[-_.]+", "-", name).lower()


def _b64(digest: bytes) -> str:
  return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _hash_file(item: tuple[str, str]) -> tuple[str | None, int]:
  path, expected = item
  try:
    with open(path, "rb") as f:
      digest = hashlib.file_digest(f, "sha256").digest()
      size = f.tell()
  except OSError as e:
    return f"{path}: cannot be read ({e.strerror})", 0
  if _b64(digest) != expected:
    return f"{path}: its sha256 is not the one RECORD lists", size
  return None, size


def _compute_seal() -> Seal:
  start = time.perf_counter()

  def unsealed(reason: str) -> Seal:
    return Seal(None, reason, {}, time.perf_counter() - start, 0, 0)

  roots = _SEAL_ROOTS
  found = (
      importlib.metadata.distributions()
      if roots is None
      else importlib.metadata.distributions(path=list(roots))
  )
  dists: dict[str, importlib.metadata.Distribution] = {}
  for dist in found:
    name = dist.metadata["Name"]
    if name is None:
      continue
    normalized = _normalize_distribution(name)
    if _SEALED_DISTRIBUTIONS.match(normalized) and normalized not in dists:
      dists[normalized] = dist
  for required in _REQUIRED_DISTRIBUTIONS:
    if required not in dists:
      return unsealed(f"no installed distribution {required!r}")

  manifest = hashlib.sha256()
  entries: list[tuple[str, str]] = []
  names: dict[str, str] = {}
  for name in sorted(dists):
    dist = dists[name]
    files = dist.files
    records = [
        f for f in files or ()
        if f.name == "RECORD" and f.parent.name.endswith(".dist-info")
    ]
    if len(records) != 1:
      return unsealed(f"{name} has no RECORD (an editable or source install)")
    try:
      record = dist.locate_file(records[0]).read_bytes()
    except OSError as e:
      return unsealed(f"{name}'s RECORD cannot be read ({e.strerror})")
    manifest.update(name.encode())
    manifest.update(b"\0")
    manifest.update(_LEN.pack(len(record)))
    manifest.update(record)
    hashed = 0
    for f in files or ():
      if f.hash is None:
        continue
      if f.hash.mode != "sha256":
        return unsealed(f"{name}'s RECORD lists {f.hash.mode} for {f}")
      path = str(dist.locate_file(f))
      entries.append((path, f.hash.value))
      names[path] = f.as_posix()
      hashed += 1
    if not hashed:
      return unsealed(f"{name}'s RECORD lists no file hashes")

  workers = min(8, os.cpu_count() or 1)
  total_bytes = 0
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    for problem, size in pool.map(_hash_file, entries):
      total_bytes += size
      if problem is not None:
        return unsealed(problem)

  sealed_files: dict[str, str] = {}
  hashes: dict[str, str] = {}
  for path, expected in entries:
    sealed_files[os.path.normpath(path)] = names[path]
    sealed_files[os.path.realpath(path)] = names[path]
    hashes[names[path]] = expected
  verified = set()
  for name, module in list(sys.modules.items()):
    verified.add(name)
    if name.partition(".")[0] not in SEALED_MODULE_ROOTS:
      continue
    file = getattr(module, "__file__", None)
    if file is None:
      continue
    if (
        os.path.normpath(file) not in sealed_files
        and os.path.realpath(file) not in sealed_files
    ):
      return unsealed(
          f"module {name} is {file}, which no installed RECORD lists"
      )
  sources, source_digests, problem = _sources(
      os.environ.get(SOURCES_VARIABLE, ""))
  if problem is not None:
    return unsealed(problem)
  return Seal(
      manifest.digest(), None, sealed_files,
      time.perf_counter() - start, len(entries), total_bytes,
      hashes, frozenset(verified), sources=sources, source_digests=source_digests,
  )


def _sources(declared: str) -> tuple[dict[str, tuple[str, str]], dict[str, bytes], str | None]:
  """The declared source packages' files and tree digests (`SOURCES_VARIABLE`).

  Each package is found where an import would find it; its every Python file is
  hashed, and its digest is the hash of its files' names and hashes in order. A
  package that is declared and not found, or a namespace package (no one tree),
  is a declaration the process cannot keep, and unseals it.
  """
  files: dict[str, tuple[str, str]] = {}
  digests: dict[str, bytes] = {}
  for package in sorted({p.strip() for p in declared.split(",") if p.strip()}):
    spec = importlib.util.find_spec(package)
    locations = list(spec.submodule_search_locations or ()) if spec else []
    if spec is None or spec.origin is None or len(locations) != 1:
      return {}, {}, (
          f"the declared source package {package!r} is no one importable tree")
    root = os.path.realpath(locations[0])
    tree = hashlib.sha256()
    for directory, subdirectories, names in os.walk(root):
      subdirectories.sort()
      for name in sorted(names):
        if not name.endswith(".py"):
          continue
        path = os.path.join(directory, name)
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        with open(path, "rb") as f:
          digest = hashlib.sha256(f.read()).digest()
        tree.update(_framed(relative.encode(), digest))
        files[os.path.normpath(path)] = (package, relative)
        files[os.path.realpath(path)] = (package, relative)
    digests[package] = tree.digest()
  return files, digests, None


_lock = threading.RLock()
_seal: Seal | None = None


def seal() -> Seal:
  """The process's code identity, computed on first use.

  What it cannot establish (a file that differs or cannot be read, no RECORD) is
  the seal's own reason, and is remembered. An error of the sealing itself is not
  one the installation has, and propagates.
  """
  global _seal
  with _lock:
    if _seal is None:
      _seal = _compute_seal()
    return _seal
