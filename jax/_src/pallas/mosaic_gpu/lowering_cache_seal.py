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
the manifests.

What runs is then known by its code, not by its name: ``Seal.code_identity`` is
the identity of a code object (the sealed file it is of, where, its name and a
hash of its bytecode, constants and names, nested code included),
``Seal.function_identity`` that of a function that runs the globals of a sealed
module, and ``Seal.class_identity`` that of a class a sealed module defines, found
under its name there, with the code of every method it has (a class of the
standard library, NumPy or ml_dtypes, whose versions are part of every key, or of
an extension module of the installation, whose bytes are sealed, is known by its
name). ``Seal.verify_loaded`` holds a module loaded after sealing to the same
RECORD.
"""

from __future__ import annotations

import base64
import concurrent.futures
import enum
import functools
import hashlib
import importlib.metadata
import os
import re
import struct
import sys
import sysconfig
import threading
import time
import types
from collections.abc import Iterator, Sequence
from typing import Any

# The distributions whose installed files are the code that lowers.
_SEALED_DISTRIBUTIONS = re.compile(r"^(jax|jaxlib|jax-cuda\d+-.+)$")
_REQUIRED_DISTRIBUTIONS = ("jax", "jaxlib")
# Whose modules must all be files of those distributions.
SEALED_MODULE_ROOTS = frozenset({"jax", "jaxlib"})

# Private: where the distributions' manifests are looked for. None is the
# interpreter's sys.path. A test sets it to a site directory it made.
_SEAL_ROOTS: Sequence[str] | None = None

# The file name of code that `exec` or `eval` makes (a dataclass's methods).
_GENERATED = "<string>"

_LEN = struct.Struct("<Q")

EXTENSION_SUFFIXES = (".so", ".pyd", ".dylib")

# `type.__flags__`: a type made by Python code (or from a spec) and a type that
# cannot have attributes set on it.
_HEAP_TYPE = 1 << 9
_IMMUTABLE_TYPE = 1 << 8

class Origin(enum.Enum):
  """Whose code a module is."""
  # A Python file of the sealed installation: its code is what RECORD hashes.
  SEALED = "sealed"
  # A compiled module of the sealed installation (or one inside it): bytes RECORD hashes.
  EXTENSION = "extension"
  # The interpreter's and its standard library: the Python version is in every key.
  STANDARD = "standard"
  # A file of another installed distribution (NumPy, ml_dtypes, ...): its version
  # is in the key, and what it is is not checked.
  THIRD_PARTY = "third party"
  # Anything else: a file of the user's, a module made at run time.
  UNKNOWN = "unknown"


class Unsealed(Exception):
  """The code this lowering runs is not the sealed installation's."""


class NotSealed(Exception):
  """A function, a class or a piece of code is not the sealed installation's."""


class NoIdentity(NotSealed):
  """A class has a member that is called, and nothing says what it does.

  Not a sign of anything made to be something else: a class that cannot be
  identified by its code (and so is not one a key can hold) may still be bound
  in a module, by its name.
  """


def _framed(*parts: bytes) -> bytes:
  return b"".join(_LEN.pack(len(p)) + p for p in parts)


class Seal:
  """The installed code's identity, or why it has none."""

  def __init__(
      self, digest: bytes | None, reason: str | None, files: dict[str, str],
      seconds: float, file_count: int, byte_count: int,
      hashes: dict[str, str] | None = None,
      verified_modules: frozenset[str] = frozenset(),
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
    # By id, with the code kept, as code objects compare by value and not by file.
    self._code_identities: dict[tuple[int, bool], tuple[types.CodeType, bytes]] = {}
    # What a module is, by name and the module the name was found to be.
    self._origins: dict[tuple[str, int], tuple[Origin, tuple[str, str] | None]] = {}
    # By id, with the class kept: what a class is, until a change is reported.
    self._class_identities: dict[int, tuple[type, bytes]] = {}

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
    if top in sys.stdlib_module_names and _in_standard_library(file):
      return Origin.STANDARD, None
    dist = _distribution_of(top, file)
    if dist is not None:
      return Origin.THIRD_PARTY, dist
    return Origin.UNKNOWN, None

  def forget_classes(self) -> None:
    """Forgets what classes were, as one was changed."""
    self._class_identities.clear()

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

  def _constant(self, value: Any, foreign: bool = False) -> bytes:
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
      return b"T" + _framed(*(self._constant(v, foreign) for v in value))
    if t is frozenset:
      return b"Z" + _framed(*sorted(self._constant(v, foreign) for v in value))
    if t is types.CodeType:
      return b"K" + self.code_identity(value, foreign=foreign)
    raise NotSealed(f"a code constant of the type {t.__qualname__} has no identity")

  def code_identity(self, code: types.CodeType, *, foreign: bool = False) -> bytes:
    """What the code object is: where in which sealed file, and what it does.

    The file it is of (relative to its distribution), the line it starts at and
    its name say which definition of the file it is; a hash of everything it
    does (bytecode, constants, the names it reads, its variables and exception
    table, nested code included) says it is the one that file defines. Code of the
    standard library or of another installed package (`foreign`) is what its
    hash says, wherever the package is installed.
    """
    memo = self._code_identities.get((id(code), foreign))
    if memo is not None and memo[0] is code:
      return memo[1]
    if foreign:
      relative = "foreign"
    else:
      relative = self.relative(code.co_filename)
    if relative is None:
      if code.co_filename != _GENERATED:
        raise NotSealed(
            f"{code.co_qualname} is defined in {code.co_filename}, which the"
            " sealed installation does not hold"
        )
      relative = _GENERATED
    body = hashlib.sha256()
    body.update(_framed(
        code.co_qualname.encode(),
        str((code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
             code.co_flags)).encode(),
        code.co_code,
        self._constant(code.co_names, foreign),
        self._constant(code.co_varnames, foreign),
        self._constant(code.co_freevars, foreign),
        self._constant(code.co_cellvars, foreign),
        code.co_exceptiontable,
        self._constant(code.co_consts, foreign),
    ))
    identity = _framed(
        relative.encode(), str(code.co_firstlineno).encode(),
        code.co_name.encode(), body.digest(),
    )
    self._code_identities[(id(code), foreign)] = (code, identity)
    return identity

  def function_identity(
      self, fn: types.FunctionType, *, generated_ok: bool = False
  ) -> bytes:
    """The identity of a function that runs the globals of a loaded module.

    A function is not what it says (its name, its module): it is the code it
    runs and the globals it runs it with, which must be the dictionary of a
    module, and the module a file of the sealed installation (the code is then
    what is hashed), of the standard library or of an installed distribution
    (then it is where that is installed, by name and version).
    `generated_ok` admits code `exec` made (the methods a dataclass gives its
    class).
    """
    code = fn.__code__
    if code.co_filename == _GENERATED:
      if not generated_ok:
        raise NotSealed(f"{code.co_qualname} is code that was generated at run time")
      return self.code_identity(code)
    name = _name_of_globals(fn.__globals__)
    module = sys.modules.get(name) if name is not None else None
    if module is None:
      raise NotSealed(
          f"{code.co_qualname} runs with the globals of no loaded module"
      )
    origin, dist = self.module_origin(name)
    if origin is Origin.SEALED:
      return self.code_identity(code)
    if origin in (Origin.STANDARD, Origin.THIRD_PARTY):
      # The standard library's and an installed package's code is what its
      # bytecode says it is, wherever it is installed: a function that is not
      # the package's own (another one that runs the package's globals) has
      # another identity, not the package's.
      return _framed(
          origin.value.encode(), name.encode(), code.co_qualname.encode(),
          *((dist[1].encode(),) if dist else ()),
          self.code_identity(code, foreign=True))
    raise NotSealed(
        f"{code.co_qualname} runs with the globals of {name}, which is not a"
        f" file of the sealed installation ({origin.value})"
    )

  # -- classes

  def class_identity(self, cls: type) -> bytes:
    """The identity of a class: its name, its bases', and what it defines.

    What a class says it is (its name, its module) is not what it is: it is the
    code of its methods, its plain data and its bases. If the module it says it is
    of is a file of the sealed installation, the code of every method it defines
    is sealed code, which identifies it. A class of the standard library or of
    an installed distribution (NumPy, ml_dtypes, ...) is identified by its name and
    its data and the code of its methods (by its hash, wherever it is installed):
    a class somebody put in a package's namespace, with other code, is another
    class, not the package's. A class that is not any of these has no identity.
    """
    memo = self._class_identities.get(id(cls))
    if memo is not None and memo[0] is cls:
      return memo[1]
    identity = _framed(*(
        self._own_class_identity(base) for base in cls.__mro__
        if base is not object
    ))
    self._class_identities[id(cls)] = (cls, identity)
    return identity

  def _own_class_identity(self, cls: type) -> bytes:
    qualname = cls.__qualname__
    if cls.__flags__ & _IMMUTABLE_TYPE and not cls.__flags__ & _HEAP_TYPE:
      # A type C code made and that cannot be changed (`int`, `types.ModuleType`,
      # NumPy's scalar types): nothing can be added to it, whatever it is called.
      return _framed(b"static", f"{cls.__module__}:{qualname}".encode())
    module = sys.modules.get(cls.__module__)
    if module is None:
      raise NotSealed(f"the class {cls.__module__}.{qualname} is of no loaded module")
    origin, dist = self.module_origin(cls.__module__)
    class_name = f"{cls.__module__}:{qualname}".encode()
    if origin is Origin.EXTENSION:
      return _framed(b"extension", class_name, *self._check_native(cls))
    if origin in (Origin.STANDARD, Origin.THIRD_PARTY):
      return _framed(
          origin.value.encode(), class_name, *self._check_provenance(cls, dist))
    if origin is not Origin.SEALED:
      raise NotSealed(
          f"the class {cls.__module__}.{qualname} is of a module that is not a"
          f" file of the sealed installation ({origin.value})"
      )
    members = []
    for name, value in sorted(vars(cls).items()):
      if name in _CLASS_METADATA:
        continue
      functions = list(_functions_of(value))
      if functions:
        for function in functions:
          members.append(_framed(
              name.encode(), self.function_identity(function, generated_ok=True)))
      elif (wrapped := _jit_wrapped(value)) is not None:
        members.append(_framed(
            name.encode(), self.function_identity(wrapped, generated_ok=True)))
      elif _is_data(value):
        members.append(_framed(name.encode(), self._constant(value)))
      elif cls in type(value).__mro__:
        continue  # A member of an enum, which the class names.
      elif _is_other_code(value):
        raise NoIdentity(
            f"{cls.__qualname__}.{name} is code ({type(value).__qualname__}) with"
            " no identity"
        )
    return _framed(class_name, *members)

  def _check_native(self, cls: type) -> tuple[bytes, ...]:
    """What a class of a compiled module is: the Python code it has (an enum a
    compiled module makes has the `enum` module's methods) by its identity."""
    members = []
    for name, value in sorted(vars(cls).items()):
      if name in _CLASS_METADATA:
        continue
      functions = list(_functions_of(value))
      wrapped = _jit_wrapped(value)
      if wrapped is not None:
        functions.append(wrapped)
      for function in functions:
        members.append(_framed(
            name.encode(), self.function_identity(function, generated_ok=True)))
      if not functions and _is_data(value):
        members.append(_framed(name.encode(), self._constant(value)))
    return tuple(members)

  def _check_provenance(
      self, cls: type, dist: tuple[str, str] | None) -> tuple[bytes, ...]:
    """What a class of the standard library or an installed package is: its
    Python code by its identity (so that a class somebody put in the package's
    namespace, with other code, is another class) and its plain data."""
    members = []
    for name, value in sorted(vars(cls).items()):
      if name in _CLASS_METADATA:
        continue
      functions = list(_functions_of(value))
      wrapped = _jit_wrapped(value)
      if wrapped is not None:
        functions.append(wrapped)
      for function in functions:
        members.append(_framed(
            name.encode(), self.function_identity(function, generated_ok=True)))
      if functions:
        continue
      if _is_data(value):
        members.append(_framed(name.encode(), self._constant(value)))
      elif cls in type(value).__mro__:
        continue  # A member of an enum, which the class names.
      elif _is_other_code(value):
        raise NoIdentity(
            f"{cls.__module__}.{cls.__qualname__}.{name} is code"
            f" ({type(value).__qualname__}) with no identity"
        )
    if dist is not None:
      members.append(_framed(dist[0].encode(), dist[1].encode()))
    return tuple(members)


# The attributes of a class that say where it is, not what it does.
_CLASS_METADATA = frozenset({
    "__module__", "__qualname__", "__doc__", "__dict__", "__weakref__",
    "__firstlineno__", "__static_attributes__",
})

_DESCRIPTORS = (
    types.WrapperDescriptorType, types.MethodDescriptorType,
    types.ClassMethodDescriptorType, types.GetSetDescriptorType,
    types.MemberDescriptorType, types.BuiltinFunctionType,
    types.BuiltinMethodType,
)


def _functions_of(value: Any) -> Iterator[types.FunctionType]:
  """The Python functions a class attribute runs."""
  if isinstance(value, types.FunctionType):
    yield value
  elif isinstance(value, (staticmethod, classmethod)):
    yield from _functions_of(value.__func__)
  elif isinstance(value, (property, types.DynamicClassAttribute)):
    for accessor in (value.fget, value.fset, value.fdel):
      if accessor is not None:
        yield from _functions_of(accessor)
  elif isinstance(value, (functools.cached_property, functools.partialmethod)):
    yield from _functions_of(value.func)


def _jit_wrapped(value: Any) -> types.FunctionType | None:
  """The function a `jax.jit`-wrapped class attribute runs, or None."""
  if type(value).__name__ == "PjitFunction":
    wrapped = getattr(value, "__wrapped__", None)
    if isinstance(wrapped, types.FunctionType):
      return wrapped
  return None


def _is_data(value: Any) -> bool:
  """Whether a class attribute is a plain constant, which a class may set."""
  t = type(value)
  if t is tuple:
    return all(_is_data(v) for v in value)
  return value is None or t in (bool, int, float, complex, str, bytes)


def _is_other_code(value: Any) -> bool:
  """Whether a class attribute runs code that is neither a function of the
  class nor a part of the interpreter."""
  return (
      callable(value) and not isinstance(value, (type, *_DESCRIPTORS))
      and type(value).__module__ != "nanobind"
  )


_GLOBALS_NAMES: dict[int, tuple[dict[str, Any], str | None]] = {}


def _name_of_globals(namespace: dict[str, Any]) -> str | None:
  """The name of the loaded module whose dictionary is `namespace`, if any.

  A module says its own name, and so does a function's globals, but a module can
  say another's (the standard library's `_collections_abc` calls itself
  `collections.abc`): the dictionary is what is compared, by identity.
  """
  said = namespace.get("__name__")
  if isinstance(said, str):
    module = sys.modules.get(said)
    if module is not None and module.__dict__ is namespace:
      return said
  found = _GLOBALS_NAMES.get(id(namespace))
  if found is not None and found[0] is namespace:
    return found[1]
  name = next(
      (n for n, m in list(sys.modules.items())
       if m is not None and getattr(m, "__dict__", None) is namespace), None)
  _GLOBALS_NAMES[id(namespace)] = (namespace, name)
  return name


def _in_standard_library(file: str) -> bool:
  standard = os.path.realpath(sysconfig.get_paths()["stdlib"]) + os.sep
  return os.path.realpath(file).startswith(standard)


def _is_standard_module(name: str, module: types.ModuleType) -> bool:
  """Whether a module is the interpreter's or the standard library's, which the
  Python version (a part of every key) names."""
  file = getattr(module, "__file__", None)
  if file is None:
    return name in sys.stdlib_module_names or name == "builtins"
  return name.partition(".")[0] in sys.stdlib_module_names and _in_standard_library(file)


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
  return Seal(
      manifest.digest(), None, sealed_files,
      time.perf_counter() - start, len(entries), total_bytes,
      hashes, frozenset(verified),
  )


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
