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

"""What every function the lowering can call is, as one identity.

The seal says the files are the installed ones. Python can still make a process
run something else than the files define: a function a module defines rebound to
another (a sealed one, even: ``lowering._bcast = lowering._bcast_wg``), a method
set on a class, a constant rebound, a primitive given another ``abstract_eval``.
A lowering reads all of these through module globals, and so does the retracing
it makes of a jaxpr (JAX's tracing core, state discharge, ...). So the universe
of what it can run is every module of the sealed installation the process has
loaded (but ``jax.experimental``'s, bar its Mosaic and Pallas packages), and
the identity of the universe is, for every module, what each name in it is:

* a *function*, by its code (see ``Seal.function_identity``), its defaults, its
  keyword defaults and what its closure holds;
* a *class*, by its methods' code (``Seal.class_identity``) and theirs
  defaults and closures;
* a *partial*, by its function, arguments and keywords;
* a *primitive*, by its name, its class and what is set on it (``impl``,
  ``abstract_eval``, ...);
* a *constant* (a number, a string, an enum member, a dtype, a frozen
  dataclass, a NumPy scalar), by its value;
* anything else (a lock, a cache, a registry the lowering does not own), by its type.

A process that rebinds any of them has another identity, so another key: the
rebinding is never answered from an entry a process without it made. Code that
is not sealed (a user's function put in a module's namespace) leaves the process
unsealed, and so does a name that was a constant and now is something that is
not one.

The identity is computed once (it hashes the code of every function, about a
second) and again for what changed. What changed is found, at every lowering, by
comparing the identity of the objects each module and class holds with what
they held (cheap, and exact for what is immutable), and by what Python reports
(``counted_registry``: a function's code or defaults assigned, an item of its
keyword defaults (a counted dictionary since it was first looked at), a module's
attribute assigned, a primitive's attribute assigned). A change that is made and
undone while a kernel is lowered is reported by the count but not by the
comparison; what Python does not report (the contents of a closure's cell, an
attribute assigned to a class, a write straight into a module's dictionary) is
compared at both ends only.
"""

from __future__ import annotations

import dataclasses
import dis
import enum
import functools
import hashlib
import operator
import sys
import threading
import types
from typing import Any

from jax._src import core as jax_core
from jax._src import counted_registry
from jax._src.pallas.mosaic_gpu import lowering_cache_encoding as encoding
from jax._src.pallas.mosaic_gpu import lowering_cache_seal as sealing
import numpy as np

_EXPERIMENTAL_IN_SCOPE = ("jax.experimental.mosaic", "jax.experimental.pallas")


# The cache itself is no part of what it keys (its code is what the seal's digest
# is of), and `counted_registry` is what counts: its epoch is a constant that
# changes with every change there is.
_EXCLUDED = (
    "jax._src.pallas.mosaic_gpu.lowering_cache", "jax._src.counted_registry",
)


def in_scope(name: str) -> bool:
  """Whether a module is part of the universe (by its name)."""
  if name.partition(".")[0] not in sealing.SEALED_MODULE_ROOTS:
    return False
  if name.startswith(_EXCLUDED):
    return False
  return not name.startswith("jax.experimental.") or name.startswith(
      _EXPERIMENTAL_IN_SCOPE)


_OPAQUE = b"o"
_MISSING = object()
_MAX_DEPTH = 3


def _cell_id(cell: types.CellType) -> int:
  try:
    return id(cell.cell_contents)
  except ValueError:
    return 0


def callable_token(value: Any) -> tuple[Any, ...]:
  """What a callable is, by the identity of what it is made of.

  Exact for what is immutable; the contents of a closure's cell, and of a
  keyword defaults dictionary, are compared by the identity of their items.
  """
  if isinstance(value, types.FunctionType):
    kwdefaults = value.__kwdefaults__
    return (
        id(value), id(value.__code__), id(value.__defaults__), id(kwdefaults),
        tuple(map(id, kwdefaults.values())) if kwdefaults else (),
        id(value.__closure__),
        tuple(_cell_id(c) for c in value.__closure__) if value.__closure__ else (),
    )
  if isinstance(value, functools.partial):
    return (
        id(value), id(value.func), id(value.args),
        tuple((k, id(v)) for k, v in value.keywords.items()),
    )
  if isinstance(value, counted_registry.FrozenPartial):
    return (id(value),)
  if isinstance(value, (staticmethod, classmethod)):
    return (id(value), callable_token(value.__func__))
  if isinstance(value, (property, functools.cached_property)):
    accessors = (getattr(value, a, None) for a in ("fget", "fset", "fdel", "func"))
    return (id(value),) + tuple(
        callable_token(f) for f in accessors if f is not None)
  return (id(value),)


_LOOP_TARGETS: dict[str, frozenset[str] | None] = {}
_TARGET_INSTRUCTIONS = frozenset({
    "UNPACK_SEQUENCE", "UNPACK_EX", "STORE_NAME", "STORE_GLOBAL", "EXTENDED_ARG",
})


def _loop_targets(module: types.ModuleType) -> frozenset[str] | None:
  """The names the `for` loops at module level of a module assign (None if its
  code cannot be read).

  A variable such a loop leaves is bound to the last element, which for a loop
  over a set is another in each process: it is no part of what the module is.
  """
  name = module.__name__
  if name not in _LOOP_TARGETS:
    targets: set[str] | None = set()
    try:
      code = module.__spec__.loader.get_code(name)  # pytype: disable=attribute-error
      following = False
      for instruction in dis.get_instructions(code):
        if instruction.opname == "FOR_ITER":
          following = True
        elif following:
          if instruction.opname not in _TARGET_INSTRUCTIONS:
            following = False
          elif instruction.opname in ("STORE_NAME", "STORE_GLOBAL"):
            targets.add(instruction.argval)
    except (AttributeError, ImportError, OSError, ValueError, TypeError):
      targets = None
    _LOOP_TARGETS[name] = frozenset(targets) if targets is not None else None
  return _LOOP_TARGETS[name]


def _is_named(name: str, value: Any, module: types.ModuleType) -> bool:
  """Whether a binding of a function or a class is part of what a module is.

  It is if the name is the one the object has. Another name for it (`_bcast =
  _bcast_wg`, `from m import f as g`) is as well, unless it is the variable of a
  `for` loop at module level, which says nothing (see `_loop_targets`). A
  binding that is not part of what a module is, is not left unchecked: the code of
  every function bound in a module is sealed code.
  """
  if name == value.__name__:
    return True
  targets = _loop_targets(module)
  return targets is not None and name not in targets


def _native_name(value: Any) -> bytes:
  """A function of a compiled module (or a builtin), by what it is called."""
  return (
      f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', '')}"
      .encode())


_CELL_CONTENTS = operator.attrgetter("cell_contents")


def _cells(closure: tuple[types.CellType, ...]) -> tuple[int, ...]:
  try:
    return tuple(map(id, map(_CELL_CONTENTS, closure)))
  except ValueError:  # An empty cell.
    return tuple(_cell_id(c) for c in closure)


def watch_token(value: Any) -> tuple[Any, ...]:
  """What a function (or partial) holds that is changed without assigning it: the
  contents of its closure's cells and of its keyword defaults, by identity. (That
  its code, defaults and keyword defaults are assigned is what Python reports.)"""
  if isinstance(value, functools.partial):
    return (id(value.func), _ids(value.keywords), tuple(map(id, value.args)))
  kwdefaults = value.__kwdefaults__
  closure = value.__closure__
  return (
      tuple(map(id, kwdefaults.values())) if kwdefaults else (),
      _cells(closure) if closure else (),
  )


def _dict_token(d: dict[str, Any]) -> tuple[tuple[int, ...], tuple[str, ...]]:
  return tuple(map(id, d.values())), tuple(d)


def _ids(d: Any) -> tuple[int, ...]:
  """What a class's or a primitive's dictionary holds, by identity."""
  return tuple(map(id, d.values()))


@dataclasses.dataclass
class _Module:
  name: str
  module: types.ModuleType
  namespace: dict[str, Any]
  token: tuple[tuple[int, ...], tuple[str, ...]]
  fingerprint: bytes
  # The classes it defines, the primitives it holds, and the callables whose
  # closures and keyword defaults have contents: each with its token.
  classes: list[tuple[type, tuple[int, ...]]]
  primitives: list[tuple[jax_core.Primitive, tuple[int, ...]]]
  watched: list[tuple[Any, tuple[Any, ...]]]


def _is_constant(value: Any) -> bool:
  t = type(value)
  if value is None or t in (bool, int, float, complex, str, bytes):
    return True
  if t in (tuple, frozenset):
    return all(_is_constant(v) for v in value)
  if isinstance(value, (enum.Enum, np.dtype, np.generic)):
    return True
  return (
      dataclasses.is_dataclass(value) and not isinstance(value, type)
      and t.__dataclass_params__.frozen
  )


class _Fingerprinter:
  """The identity of the names of one module."""

  def __init__(self, seal: sealing.Seal, tracked: set[tuple[str, str]]):
    self.seal = seal
    self.encoder = encoding.Encoder()
    self.tracked = tracked
    self.classes: list[tuple[type, tuple[int, ...]]] = []
    self._tracked_classes: set[int] = set()
    self._watched_ids: set[int] = set()
    self.primitives: list[tuple[jax_core.Primitive, tuple[int, ...]]] = []
    self.watched: list[tuple[Any, tuple[Any, ...]]] = []

  def _sealed(self, where: str, what: Any, *, generated_ok: bool = False) -> bytes:
    try:
      if isinstance(what, type):
        return self.seal.class_identity(what)
      return self.seal.function_identity(what, generated_ok=generated_ok)
    except sealing.NotSealed as e:
      raise sealing.Unsealed(f"{where}: {e}") from e

  def _value(self, where: str, value: Any, depth: int) -> bytes:
    """What a default or a closure's cell holds."""
    if isinstance(value, types.FunctionType):
      if depth >= _MAX_DEPTH:
        return b"f..."
      return b"f" + self._function(where, value, depth + 1)
    if isinstance(value, (functools.partial, counted_registry.FrozenPartial)):
      return b"p" + self._partial(where, value, depth + 1)
    if isinstance(value, type):
      return b"c" + self._class(where, value)
    if isinstance(value, types.BuiltinFunctionType) or (
        type(value).__module__ == "nanobind"):
      return b"x" + _native_name(value)
    if _is_constant(value):
      try:
        return b"k" + self.encoder.encode(value)
      except encoding.Uncacheable:
        pass
    return _OPAQUE + type(value).__qualname__.encode()

  def _function(self, where: str, fn: types.FunctionType, depth: int = 0) -> bytes:
    identity = self._sealed(where, fn, generated_ok=True)
    kwdefaults = fn.__kwdefaults__
    if type(kwdefaults) is dict and self.seal.holds(fn.__code__.co_filename):
      # The keyword defaults of a function of the sealed installation are a
      # dictionary anyone with the function can change in place: made a counted
      # one, a change to it is counted.
      kwdefaults = fn.__kwdefaults__ = counted_registry.CountedDict(
          kwdefaults, name=f"{where}.__kwdefaults__")
    defaults, closure = fn.__defaults__, fn.__closure__
    if defaults is None and kwdefaults is None and closure is None:
      return identity
    parts = [identity]
    if defaults:
      parts.append(encoding.seq(
          b"D", [self._value(where, v, depth) for v in defaults]))
    if kwdefaults:
      parts.append(encoding.seq(b"K", [
          encoding.string(k) + self._value(where, v, depth)
          for k, v in sorted(kwdefaults.items())
      ]))
    if closure:
      cells = []
      for cell in closure:
        try:
          cells.append(self._value(where, cell.cell_contents, depth))
        except ValueError:
          cells.append(b"empty")
      parts.append(encoding.seq(b"C", cells))
    if kwdefaults or closure:
      self._watch(fn)
    return encoding.seq(b"F", parts)

  def _partial(self, where: str, partial: Any, depth: int = 0) -> bytes:
    func = partial.func
    inner = (self._function(where, func, depth) if isinstance(func, types.FunctionType)
             else self._value(where, func, depth))
    parts = [
        inner,
        encoding.seq(b"A", [self._value(where, v, depth) for v in partial.args]),
        encoding.seq(b"K", [
            encoding.string(k) + self._value(where, v, depth)
            for k, v in sorted(partial.keywords.items())
        ]),
    ]
    if isinstance(partial, functools.partial):
      self._watch(partial)
    return encoding.seq(b"P", parts)

  def _class(self, where: str, cls: type) -> bytes:
    try:
      return self.seal.class_identity(cls)
    except sealing.NotSealed as e:
      if isinstance(e, sealing.NoIdentity):
        # Members this cannot identify (an object that is called, a method of a
        # class of a compiled module): the class by what it is called and made of.
        return encoding.seq(b"opaque-class", [
            encoding.string(f"{cls.__module__}:{cls.__qualname__}"),
            *(encoding.string(n) for n in sorted(vars(cls))),
        ])
      raise sealing.Unsealed(f"{where}: {e}") from e

  def _track(self, cls: type) -> None:
    """Holds a class defined here, and the classes defined in it, to compare."""
    if id(cls) in self._tracked_classes:
      return
    self._tracked_classes.add(id(cls))
    self.classes.append((cls, _ids(vars(cls))))
    for member in list(vars(cls).values()):
      if isinstance(member, type) and member.__module__ == cls.__module__:
        self._track(member)

  def _watch(self, value: Any) -> None:
    if id(value) not in self._watched_ids:
      self._watched_ids.add(id(value))
      self.watched.append((value, watch_token(value)))

  def _methods(self, where: str, cls: type) -> bytes:
    """What the methods of a class have besides code: defaults and closures."""
    parts = []
    for name, member in sorted(vars(cls).items()):
      for fn in _functions_of(member):
        if fn.__defaults__ is None and fn.__kwdefaults__ is None and fn.__closure__ is None:
          continue
        parts.append(encoding.string(name) + self._function(f"{where}.{name}", fn))
    return encoding.seq(b"M", parts)

  def _primitive(self, where: str, primitive: jax_core.Primitive) -> bytes:
    state = []
    for name, value in sorted(vars(primitive).items()):
      state.append(encoding.string(name) + self._value(f"{where}.{name}", value, 0))
    self.primitives.append((primitive, _ids(vars(primitive))))
    return encoding.seq(b"Pr", [
        encoding.string(primitive.name), self._class(where, type(primitive)),
        *state,
    ])

  def binding(self, module: types.ModuleType, name: str, value: Any) -> bytes | None:
    module_name = module.__name__
    where = f"{module_name}.{name}"
    if isinstance(value, types.ModuleType):
      return b"m" + value.__name__.encode()
    if isinstance(value, types.FunctionType):
      identity = b"f" + self._function(where, value)
      return identity if _is_named(name, value, module) else None
    if isinstance(value, type):
      if value.__module__ == module_name:
        self._track(value)
        identity = b"c" + self._class(where, value) + self._methods(where, value)
      else:
        identity = b"c" + self._class(where, value)
      return identity if _is_named(name, value, module) else None
    if isinstance(value, (functools.partial, counted_registry.FrozenPartial)):
      return b"p" + self._partial(where, value)
    if isinstance(value, jax_core.Primitive):
      return b"P" + self._primitive(where, value)
    if isinstance(value, (counted_registry.CountedDict, counted_registry.CountedSet,
                          counted_registry.CountedList)):
      return b"t" + value.name.encode()  # the tables' own identity says what is in them
    t = type(value)
    if t.__name__ == "PjitFunction":
      wrapped = getattr(value, "__wrapped__", None)
      if isinstance(wrapped, types.FunctionType):
        return b"j" + self._function(where, wrapped)
    if t.__module__ == "nanobind" or isinstance(value, types.BuiltinFunctionType):
      return b"x" + _native_name(value)
    if value is not None and _is_constant(value):
      try:
        constant = b"k" + self.encoder.encode(value)
      except encoding.Uncacheable:
        constant = None
      if constant is not None:
        self.tracked.add((module_name, name))
        return constant
    if (module_name, name) in self.tracked:
      raise sealing.Unsealed(
          f"{where} was a constant and is now a {t.__qualname__}")
    # Anything else (a lock, a cache, a name a library sets when it first needs
    # it) is state, not what the lowering is, and says nothing of it.
    return None

  def module(self, name: str, module: types.ModuleType) -> bytes:
    entries = []
    for key, value in sorted(vars(module).items()):
      if key.startswith("__") and key.endswith("__"):
        continue
      fingerprint = self.binding(module, key, value)
      if fingerprint is not None:
        entries.append(encoding.string(key) + fingerprint)
    return encoding.seq(b"module", [encoding.string(name), *entries])


def _functions_of(member: Any) -> list[types.FunctionType]:
  if isinstance(member, types.FunctionType):
    return [member]
  if isinstance(member, (staticmethod, classmethod)):
    return _functions_of(member.__func__)
  if isinstance(member, (property, functools.cached_property)):
    out = []
    for a in ("fget", "fset", "fdel", "func"):
      f = getattr(member, a, None)
      if f is not None:
        out += _functions_of(f)
    return out
  return []


class Universe:
  """The identity of every function the lowering can call (see the module)."""

  def __init__(self):
    self._lock = threading.RLock()
    self._modules: dict[str, _Module] = {}
    self._digest = b""
    self._sys_modules_count = -1
    # Names that were constants at some time (and must stay ones).
    self._tracked: set[tuple[str, str]] = set()

  # -- building

  def _build(self, seal: sealing.Seal, name: str, module: types.ModuleType) -> _Module:
    namespace = vars(module)
    counted_registry.count_module(module)
    token = _dict_token(namespace)
    fingerprinter = _Fingerprinter(seal, self._tracked)
    fingerprint = fingerprinter.module(name, module)
    return _Module(
        name, module, namespace, token, fingerprint, fingerprinter.classes,
        fingerprinter.primitives, fingerprinter.watched)

  def _new_modules(self, seal: sealing.Seal) -> bool:
    count = len(sys.modules)
    if count == self._sys_modules_count:
      return False
    changed = False
    for name, module in list(sys.modules.items()):
      if name in self._modules or module is None or not in_scope(name):
        continue
      file = getattr(module, "__file__", None)
      if (file is None or not seal.holds(file)
          or file.endswith(sealing.EXTENSION_SUFFIXES)):
        continue
      self._modules[name] = self._build(seal, name, module)
      changed = True
    self._sys_modules_count = count
    return changed

  # -- what changed

  def _scan(self, seal: sealing.Seal, forced: set[str]) -> bool:
    """Compares what each module holds with what it held, and makes again the
    identity of those that differ. True if any identity is not what it was.

    Raises Unsealed if what changed is not sealed code (a function a user made
    bound in a module, a module replaced).
    """
    dirty = set(forced)
    classes_changed = False
    loaded = sys.modules
    ids = id
    for name, m in self._modules.items():
      if loaded.get(name) is not m.module:
        raise sealing.Unsealed(f"the module {name} was replaced after sealing")
      namespace = m.namespace
      if (tuple(map(ids, namespace.values())), tuple(namespace)) != m.token:
        dirty.add(name)
      for cls, token in m.classes:
        if tuple(map(ids, cls.__dict__.values())) != token:
          dirty.add(name)
          classes_changed = True
          break
      for primitive, token in m.primitives:
        if tuple(map(ids, primitive.__dict__.values())) != token:
          dirty.add(name)
          break
      for watched, token in m.watched:
        if watch_token(watched) != token:
          dirty.add(name)
          break
    if not dirty:
      return False
    if classes_changed or forced:
      # What changed may be a method of a class, which is part of the identity of
      # the classes that have it as a base (in any module): all are made again.
      seal.forget_classes()
      dirty = set(self._modules)
    changed = False
    for name in dirty:
      old = self._modules[name]
      new = self._build(seal, name, old.module)
      # What a module holds that is no part of what it is (a cache a library
      # fills) changes its dictionary and not its identity.
      self._modules[name] = new
      changed = changed or new.fingerprint != old.fingerprint
    return changed

  def tokens_unchanged(self) -> bool:
    """Whether the universe is what its identity was made of.

    What changed that is not part of its identity is not a change.
    """
    with self._lock:
      try:
        changed = self._scan(sealing.seal(), self._assigned())
      except sealing.Unsealed:
        return False
      if changed:
        self._digest = b""
      return not changed

  def _assigned(self) -> set[str]:
    """The modules of the functions whose code, defaults or keyword defaults
    were assigned since they were last looked at."""
    forced: set[str] = set()
    for fn in counted_registry.take_assigned_functions():
      owner = fn.__globals__.get("__name__")
      if owner in self._modules:
        forced.add(owner)
    return forced

  def check(self, seal: sealing.Seal) -> bytes:
    """The identity of the universe as it is now.

    Raises Unsealed if a function is not sealed code, a module was replaced or a
    constant is not one any more.
    """
    with self._lock:
      counted_registry.install_function_audit()
      assigned = self._assigned()
      changed = self._new_modules(seal)
      # (What looking at a module assigned to its functions is not a change.)
      counted_registry.take_assigned_functions()
      changed = self._scan(seal, assigned) or changed
      if changed or not self._digest:
        digest = hashlib.sha256()
        for name in sorted(self._modules):
          digest.update(self._modules[name].fingerprint)
        self._digest = digest.digest()
      return self._digest

  @property
  def module_count(self) -> int:
    return len(self._modules)


_universe = Universe()


def check(seal: sealing.Seal) -> bytes:
  """The identity of the universe, as it is now. See `Universe.check`."""
  return _universe.check(seal)


def tokens_unchanged() -> bool:
  """Whether the universe is what its identity was made of. See `Universe.tokens_unchanged`."""
  return _universe.tokens_unchanged()
