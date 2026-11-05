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

"""State that counts its mutations, so that a reader can tell it did not change.

A cache of what code produced can know what the code was only if it knows that
what the code depends on did not change while it ran. Python lets anyone change
almost anything, at any time and from any thread, so this module makes the state
a lowering depends on say when it is changed, in one monotonic ``epoch``. A
reader that sees the same epoch before and after has read state that did not
change (through the channels below), however the state moved in between:

* ``CountedDict``, ``CountedSet`` and ``CountedList`` are containers that count
  each mutation made to them. They count it before it is made and again after,
  under a lock, so a mutation in progress is odd in its ``generation``. They also
  freeze the ``functools.partial`` objects they are given (``FrozenPartial``): a
  partial's keyword dictionary can be changed in place, and a rule held by one
  could not be told from the rule it was.
* ``count_module`` makes the assignment of a module's attributes count, which is
  how a function or a constant a module defines is rebound. (A function's keyword
  defaults are a ``CountedDict`` too, which the reader of a function makes of them.)
* ``install_function_audit`` counts the assignment of a function's ``__code__``,
  ``__defaults__`` and ``__kwdefaults__``, which CPython reports to audit hooks,
  and keeps the functions they were assigned to.
* ``jax._src.core.Primitive`` counts the assignment of its attributes
  (``def_impl``, ``def_abstract_eval``, ...) with ``bump``.

What this cannot see is what Python does not report: the contents of a closure's
cell, of a default value that is a container, an attribute assigned to a class,
or a write straight into a module's ``__dict__``. A reader that wants those has to
compare them at both ends of what it reads, which does not see a change that is
made and undone in between.

Each container has a name, which is how a reader finds the one its module
defines and tells it from one that was put in its place (``named``).
"""

from __future__ import annotations

import functools
import sys
import threading
import types
from collections.abc import Callable, Mapping
from typing import Any

_named: dict[str, CountedDict | CountedSet | CountedList] = {}

# The count of every mutation made to what counts them. Only ever increases; a
# lost increment (two threads racing on it) still leaves it different.
_epoch = 0
_lock = threading.Lock()


def epoch() -> int:
  """The number of mutations counted so far, which only ever increases."""
  return _epoch


def bump() -> None:
  """Counts a mutation (made or about to be)."""
  global _epoch
  _epoch += 1


def named(name: str) -> CountedDict | CountedSet | CountedList | None:
  """The container defined under `name`, if there is one."""
  return _named.get(name)


class FrozenPartial:
  """A ``functools.partial`` whose function, arguments and keywords cannot change.

  A partial has a dictionary of keywords that anyone with the partial can change
  in place, and what it calls changes with it. This one has the same ``func``,
  ``args`` and ``keywords`` (a read-only mapping) and is called the same way.
  """

  __slots__ = ("func", "args", "_keywords")

  func: Callable[..., Any]
  args: tuple[Any, ...]
  _keywords: Mapping[str, Any]

  def __init__(self, func, args=(), keywords=None):
    object.__setattr__(self, "func", func)
    object.__setattr__(self, "args", tuple(args))
    object.__setattr__(
        self, "_keywords", types.MappingProxyType(dict(keywords or {})))

  @classmethod
  def of(cls, partial: functools.partial) -> FrozenPartial:
    return cls(partial.func, partial.args, partial.keywords)

  @property
  def keywords(self) -> Mapping[str, Any]:
    return self._keywords

  def __setattr__(self, name, value):
    raise AttributeError(f"a FrozenPartial's {name} cannot be set")

  def __delattr__(self, name):
    raise AttributeError(f"a FrozenPartial's {name} cannot be deleted")

  def __call__(self, *args, **kwargs):
    return self.func(*self.args, *args, **{**self._keywords, **kwargs})

  def __repr__(self):
    parts = [repr(self.func), *map(repr, self.args)]
    parts += [f"{k}={v!r}" for k, v in self._keywords.items()]
    return f"FrozenPartial({', '.join(parts)})"

  def __reduce__(self):
    return (FrozenPartial, (self.func, self.args, dict(self._keywords)))


def _freeze(value: Any) -> Any:
  return FrozenPartial.of(value) if type(value) is functools.partial else value


def _mutates[F: Callable[..., Any]](method: F) -> F:
  @functools.wraps(method)
  def wrapper(self, *args, **kwargs):
    with self._lock:
      self._generation += 1
      bump()
      try:
        return method(self, *args, **kwargs)
      finally:
        bump()
        self._generation += 1
  return wrapper  # type: ignore[return-value]


class CountedDict(dict):
  """A dict that counts the mutations made to it.

  Its generation is odd while a mutation is in progress and two more after one.
  A ``functools.partial`` set as a value is kept as a ``FrozenPartial``.
  """

  __slots__ = ("name", "_generation", "_lock")

  def __init__(self, *args, name: str, **kwargs):
    super().__init__(*args, **kwargs)
    for key, value in list(super().items()):
      if type(value) is functools.partial:
        super().__setitem__(key, _freeze(value))
    self.name = name
    self._generation = 0
    self._lock = threading.RLock()
    _named[name] = self

  @property
  def generation(self) -> int:
    return self._generation

  @_mutates
  def __setitem__(self, key, value):
    super().__setitem__(key, _freeze(value))

  @_mutates
  def __delitem__(self, key):
    super().__delitem__(key)

  @_mutates
  def __ior__(self, other):
    return super().__ior__(
        {k: _freeze(v) for k, v in dict(other).items()})

  @_mutates
  def clear(self):
    super().clear()

  @_mutates
  def pop(self, *args):
    return super().pop(*args)

  @_mutates
  def popitem(self):
    return super().popitem()

  @_mutates
  def setdefault(self, key, default=None):
    return super().setdefault(key, _freeze(default))

  @_mutates
  def update(self, *args, **kwargs):
    super().update({k: _freeze(v) for k, v in dict(*args, **kwargs).items()})


class CountedSet(set):
  """A set that counts the mutations made to it."""

  __slots__ = ("name", "_generation", "_lock")

  def __init__(self, *args, name: str):
    super().__init__(*args)
    self.name = name
    self._generation = 0
    self._lock = threading.RLock()
    _named[name] = self

  @property
  def generation(self) -> int:
    return self._generation

  @_mutates
  def add(self, element):
    super().add(element)

  @_mutates
  def discard(self, element):
    super().discard(element)

  @_mutates
  def remove(self, element):
    super().remove(element)

  @_mutates
  def pop(self):
    return super().pop()

  @_mutates
  def clear(self):
    super().clear()

  @_mutates
  def update(self, *others):
    super().update(*others)

  @_mutates
  def difference_update(self, *others):
    super().difference_update(*others)

  @_mutates
  def intersection_update(self, *others):
    super().intersection_update(*others)

  @_mutates
  def symmetric_difference_update(self, other):
    super().symmetric_difference_update(other)

  @_mutates
  def __ior__(self, other):
    return super().__ior__(other)

  @_mutates
  def __iand__(self, other):
    return super().__iand__(other)

  @_mutates
  def __isub__(self, other):
    return super().__isub__(other)

  @_mutates
  def __ixor__(self, other):
    return super().__ixor__(other)


class CountedList(list):
  """A list that counts the mutations made to it."""

  __slots__ = ("name", "_generation", "_lock")

  def __init__(self, *args, name: str):
    super().__init__(*args)
    self.name = name
    self._generation = 0
    self._lock = threading.RLock()
    _named[name] = self

  @property
  def generation(self) -> int:
    return self._generation

  @_mutates
  def __setitem__(self, index, value):
    super().__setitem__(index, value)

  @_mutates
  def __delitem__(self, index):
    super().__delitem__(index)

  @_mutates
  def __iadd__(self, other):
    return super().__iadd__(other)

  @_mutates
  def __imul__(self, other):
    return super().__imul__(other)

  @_mutates
  def append(self, element):
    super().append(element)

  @_mutates
  def extend(self, elements):
    super().extend(elements)

  @_mutates
  def insert(self, index, element):
    super().insert(index, element)

  @_mutates
  def remove(self, element):
    super().remove(element)

  @_mutates
  def pop(self, *args):
    return super().pop(*args)

  @_mutates
  def clear(self):
    super().clear()

  @_mutates
  def sort(self, *args, **kwargs):
    super().sort(*args, **kwargs)

  @_mutates
  def reverse(self):
    super().reverse()


# -- modules


class _CountedModule(types.ModuleType):
  """A module that counts the attributes assigned to or deleted from it."""

  def __setattr__(self, name, value):
    bump()
    super().__setattr__(name, value)
    bump()

  def __delattr__(self, name):
    bump()
    super().__delattr__(name)
    bump()


def count_module(module: types.ModuleType) -> bool:
  """Makes the assignment of `module`'s attributes count. False if it cannot be.

  Only a plain module can be (a module of a class of its own is left as it is).
  A write straight into the module's dictionary is not an assignment.
  """
  if type(module) is _CountedModule:
    return True
  if type(module) is not types.ModuleType:
    return False
  module.__class__ = _CountedModule
  return True


# -- functions

# The functions whose code, defaults or keyword defaults were assigned since they
# were last taken. Appended to by the audit hook, taken by `take_assigned_functions`.
_assigned_functions: list[types.FunctionType] = []
_audit_installed = False


def install_function_audit() -> None:
  """Counts the assignment of any function's ``__code__``, ``__defaults__`` and ``__kwdefaults__``.

  CPython reports these to audit hooks (as ``object.__setattr__``) before it makes
  them, in C, with no Python code in between. The hook cannot be removed, and
  does nothing for any other event.
  """
  global _audit_installed
  with _lock:
    if _audit_installed:
      return
    function_type = types.FunctionType
    assigned = _assigned_functions

    def hook(event, args):
      global _epoch
      if event == "object.__setattr__" and type(args[0]) is function_type:
        assigned.append(args[0])
        _epoch += 1

    sys.addaudithook(hook)
    _audit_installed = True


def take_assigned_functions() -> list[types.FunctionType]:
  """The functions assigned to since the last call (and forgets them)."""
  taken = _assigned_functions[:]
  del _assigned_functions[:len(taken)]
  return taken
