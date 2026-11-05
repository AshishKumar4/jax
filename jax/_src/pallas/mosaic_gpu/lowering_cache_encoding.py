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

"""The canonical encoding of the values a lowering's key is made of.

The encoder is strict: anything it cannot encode exactly (a class outside the
sealed installation, a mutable container, an object it has no encoding for) raises
``Uncacheable``, and the lowering is made afresh.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import functools
import re
import struct
import sys
import threading
import types
from collections.abc import Callable, Sequence
from typing import Any

import jax
from jax._src import core as jax_core
from jax._src import counted_registry
from jax._src import effects as jax_effects
from jax._src import flattree
from jax._src import frozen_dict
from jax._src import literals as jax_literals
from jax._src import mesh as mesh_lib
from jax._src import named_sharding
from jax._src import partition_spec
from jax._src import tree_util
from jax._src.lax import eval_jaxpr as eval_jaxpr_lib
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering_cache_seal as sealing
from jax._src.state import types as state_types
import numpy as np

_LEN = struct.Struct("<Q")
_MAX_DEPTH = 400


class _PrimitiveHolders:
  """Which sealed module attribute holds each primitive: its registration identity.

  A primitive is an object, and a name and a rule table do not say which: two
  primitives named "add" can both have sealed rules, which differ. The primitive
  a sealed installation has is the one a sealed module holds under a name
  (``jax._src.lax.lax.add_p``); one that no sealed module holds (a primitive a
  process made) has no identity a key could name, and its kernels are not cached.
  A primitive held under several names (re-exported) is identified by the first
  of the ``jax._src`` modules' by name, which every process that has the primitive
  has loaded.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._by_id: dict[int, tuple[str, str, jax_core.Primitive]] = {}
    self._module_count = -1

  def _scan(self) -> None:
    seal = sealing.seal()
    holders: dict[int, list[tuple[str, str, jax_core.Primitive]]] = {}
    for module_name, module in list(sys.modules.items()):
      if module_name.partition(".")[0] not in sealing.SEALED_MODULE_ROOTS:
        continue
      if module is None or not seal.holds(getattr(module, "__file__", None)):
        continue
      for attribute, value in list(vars(module).items()):
        if isinstance(value, jax_core.Primitive):
          holders.setdefault(id(value), []).append((module_name, attribute, value))
    self._by_id = {
        i: min(held, key=lambda h: (not h[0].startswith("jax._src"), h[0], h[1]))
        for i, held in holders.items()
    }
    self._module_count = len(sys.modules)

  def identity(self, primitive: jax_core.Primitive) -> tuple[str, str] | None:
    """The (module, attribute) of the sealed module that holds `primitive`."""
    with self._lock:
      held = self._by_id.get(id(primitive))
      if held is not None:
        # Still held by what held it (a module attribute can be rebound).
        module = sys.modules.get(held[0])
        if getattr(module, held[1], None) is not primitive:
          held = None
      # A primitive that is not found is looked for again only if modules were
      # loaded since the last look (one made by an import is held by its module).
      if held is None and (
          len(sys.modules) != self._module_count
          or id(primitive) in self._by_id):
        self._scan()
        held = self._by_id.get(id(primitive))
      return None if held is None else (held[0], held[1])


_HOLDERS = _PrimitiveHolders()


class Uncacheable(Exception):
  """This lowering cannot be keyed exactly: it is lowered afresh."""


def frame(tag: bytes, payload: bytes) -> bytes:
  return tag + _LEN.pack(len(payload)) + payload


def seq(tag: bytes, items: Sequence[bytes]) -> bytes:
  return tag + _LEN.pack(len(items)) + b"".join(items)


def string(s: str) -> bytes:
  return frame(b"S", s.encode("utf-8", "surrogatepass"))


def _python_class(cls: type) -> bool:
  """Whether `cls` and its bases are defined by Python files."""
  for base in cls.__mro__:
    if base is object:
      continue
    module = sys.modules.get(base.__module__)
    file = getattr(module, "__file__", None)
    if not file or not file.endswith(".py"):
      return False
  return True


def _stateless(x: Any) -> bool:
  """Whether `x` is an instance of a Python class and nothing more."""
  cls = type(x)
  if not _python_class(cls) or getattr(x, "__dict__", None):
    return False
  return all(
      getattr(base, "__slots__", ()) in ((), None, "") for base in cls.__mro__
      if base is not object
  )


class Encoder:
  """The canonical encoding of the values a lowering's key is made of.

  `Encoder.encode` raises `Uncacheable` for anything it cannot encode exactly:
  a value of a class outside the sealed installation, a mutable container, an
  object it has no encoding for. It records the primitives of the jaxprs it
  encodes.
  """

  def __init__(self):
    self.primitives: dict[jax_core.Primitive, bool] = {}
    self._callables: list[Any] = []
    # The bare `object()`s met (the axis name `warp_map` makes), by id, and the
    # objects, which keep each id from being another's.
    self._anonymous: dict[int, int] = {}
    self._anonymous_objects: list[object] = []
    self._class_identities: dict[type, bytes] = {}

  # -- entry points

  def encode(self, x: Any) -> bytes:
    return self._e(x, 0)

  # -- the dispatch

  def _e(self, x: Any, depth: int) -> bytes:
    if depth > _MAX_DEPTH:
      raise Uncacheable("a value nests too deeply to encode")
    t = type(x)
    if x is None:
      return b"N"
    if t is bool:
      return b"B1" if x else b"B0"
    if t is int:
      return frame(b"I", str(x).encode())
    if t is float:
      return b"F" + struct.pack("<d", x)
    if t is complex:
      return b"C" + struct.pack("<dd", x.real, x.imag)
    if t is str:
      return string(x)
    if t is bytes:
      return frame(b"Y", x)
    if t is tuple:
      return seq(b"T", [self._e(i, depth + 1) for i in x])
    if t is list:
      return seq(b"L", [self._e(i, depth + 1) for i in x])
    if t is dict:
      return self._mapping(b"M", x.items(), depth)
    if t is frozenset:
      return seq(b"Z", sorted(self._e(i, depth + 1) for i in x))
    if t is slice:
      return seq(b"s", [self._e(i, depth + 1) for i in (x.start, x.stop, x.step)])
    if x is Ellipsis:
      return b"E"
    if t is collections.OrderedDict:
      return seq(b"O", [
          self._e(k, depth + 1) + self._e(v, depth + 1) for k, v in x.items()
      ])
    if isinstance(x, enum.Enum):
      return self._enum(x, depth)
    if isinstance(x, np.dtype):
      return self._np_dtype(x)
    if isinstance(x, (np.generic, np.ndarray)):
      return self._numpy_value(x)
    if t in jax_literals.typed_scalar_types:
      return seq(b"J", [
          self._class(t, "typed scalar class"), self._np_dtype(x.dtype),
          self._e({int: int, float: float, complex: complex}[
              t.__mro__[1]](x), depth + 1),
      ])
    if isinstance(x, type):
      return self._type(x)
    if isinstance(x, jax_core.Jaxpr):
      return self._jaxpr(x, depth)
    if isinstance(x, tree_util.PyTreeDef):
      return self._treedef(x, depth)
    if isinstance(x, jax_core.Tracer):
      # A tracer a jaxpr holds as a constant stands for a value of an enclosing
      # jaxpr (a Ref run_scoped is given), which a lowering is handed as an
      # operand: the aval is all it can read of it.
      return seq(b"tr", [self._aval(x.aval, depth + 1)])
    if isinstance(x, jax.Array):
      value = np.asarray(x)
      return seq(b"jx", [
          self._numpy_value(value), self._e(bool(jax_core.typeof(x).weak_type), depth + 1)
      ])
    aval_encoder = _AVAL_ENCODERS.get(t)
    if aval_encoder is not None:
      return aval_encoder(self, x, depth)
    if isinstance(x, jax_core.AbstractValue):
      raise Uncacheable(
          f"the aval class {t.__module__}.{t.__qualname__} has no encoder"
      )
    if isinstance(x, jax_core.Primitive):
      return seq(b"P", [
          string(x.name), self._e(bool(x.multiple_results), depth + 1),
          self._class(t, "primitive class"), self._holder(x),
      ])
    if t is named_sharding.NamedSharding:
      return self._sharding(x, depth)
    if t is partition_spec.P:
      return seq(b"p", [
          self._e(x._partitions, depth + 1), self._e(x.unreduced, depth + 1),
          self._e(x.reduced, depth + 1), self._e(x.unreduced_kind, depth + 1),
      ])
    if t is jax_core.ManualAxisType:
      return seq(b"vma", [
          self._e(x.varying, depth + 1), self._e(x.unreduced, depth + 1),
          self._e(x.reduced, depth + 1), self._e(x.unreduced_kind, depth + 1),
      ])
    if isinstance(x, (mesh_lib.Mesh, mesh_lib.AbstractMesh)):
      return self._mesh(x, depth)
    if isinstance(x, jax_core.AxisEnv):
      return seq(b"ax", [
          self._e(tuple(x.axis_sizes.items()), depth + 1),
          self._e(frozenset(x.spmd_axis_names), depth + 1),
          self._e(frozenset(x.explicit_mesh_axis_names), depth + 1),
      ])
    if isinstance(x, frozen_dict.FrozenDict):
      return seq(b"X", [
          self._class(t, "mapping class"), self._mapping(b"M", x.items(), depth)
      ])
    if isinstance(x, flattree.FlatTree):
      return self._flat_tree(x, depth)
    if isinstance(x, types.ModuleType):
      if not sealing.seal().holds(getattr(x, "__file__", None)):
        raise Uncacheable(f"module {x.__name__} is not a sealed file")
      return frame(b"V", x.__name__.encode())
    if t.__module__ == "nanobind" and t.__name__ == "nb_func":
      return self._extension_function(x)
    if t.__name__ == "PjitFunction":
      wrapped = getattr(x, "__wrapped__", None)
      if not isinstance(wrapped, types.FunctionType):
        raise Uncacheable("a jitted function that wraps no Python function")
      return seq(b"Fj", [self._e(wrapped, depth + 1)])
    if isinstance(x, (types.FunctionType, functools.partial,
                      counted_registry.FrozenPartial, types.MethodType,
                      types.BuiltinFunctionType)):
      return self._callable(x, depth)
    if dataclasses.is_dataclass(x):
      return self._dataclass(x, depth)
    if isinstance(x, tuple) and hasattr(x, "_fields"):
      return seq(b"Q", [self._class(t, "named tuple class")] + [
          self._e(i, depth + 1) for i in x
      ])
    if t is object:
      # An anonymous sentinel: what it is, is which of the other values it
      # equals, so it is named by the order it is first met.
      n = self._anonymous.get(id(x))
      if n is None:
        n = self._anonymous[id(x)] = len(self._anonymous)
        self._anonymous_objects.append(x)
      return frame(b"Ob", str(n).encode())
    if _stateless(x):
      # A sealed class's instance with no state (jax's UNSPECIFIED, AutoLayout
      # and other singletons): the class is all it is.
      return frame(b"W", self._class(t, "class"))
    raise Uncacheable(f"{t.__module__}.{t.__qualname__} has no encoding")

  # -- helpers

  def _class(self, cls: type, what: str) -> bytes:
    """A class a sealed module defines, by what it is (see `Seal.class_identity`)."""
    identity = self._class_identities.get(cls)
    if identity is None:
      try:
        identity = sealing.seal().class_identity(cls)
      except sealing.NotSealed as e:
        raise Uncacheable(
            f"{what} {cls.__module__}.{cls.__qualname__}: {e}"
        ) from e
      self._class_identities[cls] = identity
    return frame(b"K", identity)

  def _mapping(self, tag: bytes, items, depth: int) -> bytes:
    encoded = [(self._e(k, depth + 1), self._e(v, depth + 1)) for k, v in items]
    encoded.sort(key=lambda kv: kv[0])
    return seq(tag, [k + v for k, v in encoded])

  def _enum(self, x: enum.Enum, depth: int) -> bytes:
    cls = type(x)
    # A member of a combined flag has no name: its value is what names it.
    member = string(x.name) if x.name is not None else self._e(x.value, depth + 1)
    return seq(b"U", [self._class(cls, "enum"), member])

  def _np_dtype(self, dt: np.dtype) -> bytes:
    if dt.subdtype is not None or dt.kind == "O":
      raise Uncacheable(f"the dtype {dt} has no encoding")
    if dt.fields is not None:
      # A structured dtype (jax's float0): its fields, in order.
      return seq(b"Ds", [
          string(str(dt.descr)), self._e(int(dt.itemsize), 1)
      ])
    return seq(b"D", [
        string(dt.name), string(dt.str), string(dt.kind), self._e(int(dt.itemsize), 1)
    ])

  def _numpy_value(self, x) -> bytes:
    if isinstance(x, np.ndarray):
      if x.dtype.kind == "O":
        raise Uncacheable("an object array has no encoding")
      cls = type(x)
      if cls is not np.ndarray and cls is not jax_literals.TypedNdArray:
        raise Uncacheable(f"the array class {cls.__qualname__} has no encoding")
      weak = bool(getattr(x, "aval", None) is not None and x.aval.weak_type) \
          if cls is jax_literals.TypedNdArray else False
      array = np.ascontiguousarray(x)
      return seq(b"A", [
          frame(b"K", f"{cls.__module__}:{cls.__qualname__}".encode()),
          self._np_dtype(array.dtype),
          self._e(tuple(int(d) for d in array.shape), 1),
          frame(b"Y", array.tobytes()), self._e(weak, 1),
      ])
    if x.dtype.kind == "O":
      raise Uncacheable("an object scalar has no encoding")
    return seq(b"G", [self._np_dtype(x.dtype), frame(b"Y", x.tobytes())])

  def _type(self, x: type) -> bytes:
    if issubclass(x, np.generic):
      return seq(b"Kn", [self._np_dtype(np.dtype(x))])
    if x.__module__ == "builtins" and x in (
        bool, int, float, complex, str, bytes, type(None), tuple, list, dict,
        frozenset):
      return frame(b"Kb", x.__name__.encode())
    return self._class(x, "class")

  def _dataclass(self, x: Any, depth: int) -> bytes:
    cls = type(x)
    if isinstance(x, type):
      raise Uncacheable("a dataclass type has no instance encoding")
    if not cls.__dataclass_params__.frozen:
      raise Uncacheable(
          f"the dataclass {cls.__module__}.{cls.__qualname__} is not frozen"
      )
    # Every field, whether or not it takes part in the dataclass's comparison.
    return seq(b"Dc", [self._class(cls, "dataclass")] + [
        string(f.name) + self._e(getattr(x, f.name), depth + 1)
        for f in dataclasses.fields(x)
    ])

  def _flat_tree(self, x: flattree.FlatTree, depth: int) -> bytes:
    cls = type(x)
    name = self._class(cls, "flat tree class")
    if cls is flattree.FTTuple:
      return seq(b"Ft", [name, self._e(tuple(x.elts), depth + 1)])
    if cls is flattree.FTDict:
      return seq(b"Ft", [
          name, self._e(tuple(x.keys), depth + 1), self._e(tuple(x._vals), depth + 1)
      ])
    if cls is flattree.FTPyTree:
      return seq(b"Ft", [
          name, self._e(tuple(x.xs), depth + 1), self._e(x._tree, depth + 1)
      ])
    if cls in (flattree.FTSingleton, flattree.FTStatic):
      return seq(b"Ft", [name, self._e(x.val, depth + 1)])
    raise Uncacheable(f"the flat tree class {cls.__qualname__} has no encoding")

  def _treedef(self, treedef: tree_util.PyTreeDef, depth: int) -> bytes:
    if depth > _MAX_DEPTH:
      raise Uncacheable("a tree nests too deeply to encode")
    node = treedef.node_data()
    if node is None:
      return b"Tl"
    node_type, aux = node
    if node_type in (tuple, list, dict, type(None), collections.OrderedDict):
      type_enc = frame(b"Kb", node_type.__name__.encode())
    else:
      type_enc = self._class(node_type, "pytree node type")
    return seq(b"Tn", [
        type_enc, self._e(aux, depth + 1),
        seq(b"Tc", [self._treedef(c, depth + 1) for c in treedef.children()]),
    ])

  # -- callables

  def _extension_function(self, fn: Any) -> bytes:
    """A function of an extension module of the sealed installation, by name.

    Its code is in a file RECORD hashes; which function it is, is the name its
    module (a submodule of the extension, or the extension) has it under.
    """
    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str):
      raise Uncacheable("an extension function that does not say what it is")
    parts = module_name.split(".")
    for i in range(len(parts), 0, -1):
      root = sys.modules.get(".".join(parts[:i]))
      if root is not None and getattr(root, "__file__", None):
        break
    else:
      raise Uncacheable(f"the module {module_name} of {qualname} is not loaded")
    if not sealing.seal().holds(root.__file__):
      raise Uncacheable(
          f"{module_name}.{qualname} is of {root.__file__}, which the sealed"
          " installation does not hold")
    found = root
    for part in [*parts[i:], *qualname.split(".")]:
      found = getattr(found, part, None)
    if found is not fn:
      raise Uncacheable(
          f"{module_name}.{qualname} is not the object that module has by that name")
    return seq(b"Fx", [string(module_name), string(qualname)])

  def _callable(self, fn: Any, depth: int) -> bytes:
    """A callable the sealed installation defines, with what it closes over."""
    if any(fn is c for c in self._callables):
      return b"Fcycle"
    self._callables.append(fn)
    try:
      if isinstance(fn, (functools.partial, counted_registry.FrozenPartial)):
        return seq(b"Fp", [
            self._e(fn.func, depth + 1), self._e(tuple(fn.args), depth + 1),
            self._mapping(b"M", fn.keywords.items(), depth),
        ])
      if isinstance(fn, types.MethodType):
        return seq(b"Fm", [
            self._e(fn.__func__, depth + 1), self._e(fn.__self__, depth + 1)
        ])
      if isinstance(fn, types.BuiltinFunctionType):
        owner = fn.__self__
        if isinstance(owner, types.ModuleType) and (
            owner.__name__ in sys.stdlib_module_names
            or owner.__name__ == "builtins"
        ):
          return seq(b"Fb", [string(owner.__name__), string(fn.__qualname__)])
        raise Uncacheable(
            f"the builtin {fn.__qualname__} is not a standard library function"
        )
      try:
        identity = sealing.seal().function_identity(fn)
      except sealing.NotSealed as e:
        raise Uncacheable(str(e)) from e
      cells = []
      for cell in fn.__closure__ or ():
        try:
          contents = cell.cell_contents
        except ValueError:
          cells.append(b"Fempty")
          continue
        cells.append(self._e(contents, depth + 1))
      return seq(b"Ff", [
          identity,
          seq(b"Fc", cells),
          self._e(fn.__defaults__, depth + 1),
          # (A counted dictionary is what a function's keyword defaults become
          # when `lowering_cache_universe` has looked at the function: the same.)
          self._e(
              dict(fn.__kwdefaults__)
              if isinstance(fn.__kwdefaults__, counted_registry.CountedDict)
              else fn.__kwdefaults__, depth + 1),
      ])
    finally:
      self._callables.pop()

  # -- jaxprs

  def _debug_info(self, info: Any, depth: int) -> bytes:
    if info is None:
      return b"N"
    source = info.func_src_info
    head, sep, tail = source.rpartition(" at ")
    if sep and re.fullmatch(r".*:\d+", tail):
      # "f at path:line": the path and line are a location, which no lowering
      # reads and no key holds.
      source = head
    return seq(b"Di", [
        string(info.func_name), self._e(info.arg_names, depth + 1), string(source),
    ])

  def _jaxpr(self, jaxpr: jax_core.Jaxpr, depth: int) -> bytes:
    env: dict[jax_core.Var, int] = {}

    def bind(v: jax_core.Var) -> bytes:
      env[v] = len(env)
      drop = b"d" if isinstance(v, jax_core.DropVar) else b"v"
      return drop + _LEN.pack(env[v]) + self._aval(v.aval, depth + 1)

    def atom(a: Any) -> bytes:
      if isinstance(a, jax_core.Literal):
        return seq(b"l", [self._aval(a.aval, depth + 1), self._e(a.val, depth + 1)])
      if a in env:
        return b"r" + _LEN.pack(env[a])
      raise Uncacheable("a jaxpr reads a variable it does not bind")

    def effect(e: Any) -> bytes:
      cls = type(e)
      name = self._class(cls, "effect class")
      if isinstance(e, jax_effects.JaxprInputEffect):
        i = e.input
        if isinstance(i, jax_core.Var):
          # A variable of an enclosing jaxpr, or this one's.
          i_enc = b"r" + _LEN.pack(env[i]) if i in env else seq(
              b"ev", [self._aval(i.aval, depth + 1)])
        else:
          i_enc = self._e(i, depth + 1)
        return seq(b"Ef", [name, i_enc])
      if dataclasses.is_dataclass(e):
        return seq(b"Ef", [name, self._dataclass(e, depth + 1)])
      if _stateless(e):
        return seq(b"Ef", [name])
      if _python_class(cls) and not any(
          getattr(base, "__slots__", ()) not in ((), None, "")
          for base in cls.__mro__ if base is not object
      ):
        # An effect that holds its state in its attributes (a named axis's).
        return seq(b"Ef", [
            name, self._mapping(b"M", vars(e).items(), depth + 1)
        ])
      raise Uncacheable(f"the effect {cls.__qualname__} has no encoding")

    parts = [b"jaxpr", self._debug_info(jaxpr.debug_info, depth)]
    binders = [bind(v) for v in jaxpr.all_invars]
    parts.append(seq(b"in", binders))
    parts.append(self._e(len(jaxpr.consts), depth + 1))
    parts.append(seq(b"co", [self._e(c, depth + 1) for c in jaxpr.consts]))
    eqns = []
    for eqn in jaxpr.eqns:
      prim = self._primitive(eqn.primitive, depth)
      invars = seq(b"ei", [atom(a) for a in eqn.invars])
      try:
        params = self._mapping(b"pa", eqn.params.items(), depth)
      except Uncacheable as e:
        if " (in the params of " in str(e):
          raise
        raise Uncacheable(
            f"{e} (in the params of {eqn.primitive.name})"
        ) from e
      outvars = seq(b"eo", [bind(v) for v in eqn.outvars])
      effects = seq(b"ee", sorted(effect(e) for e in eqn.effects))
      eqns.append(seq(b"eq", [prim, invars, params, outvars, effects]))
    parts.append(seq(b"eqns", eqns))
    parts.append(seq(b"out", [atom(a) for a in jaxpr.outvars]))
    parts.append(seq(b"ef", sorted(effect(e) for e in jaxpr.effects)))
    parts.append(self._e(bool(jaxpr.is_high), depth + 1))
    return seq(b"J", parts)

  def _holder(self, prim: jax_core.Primitive) -> bytes:
    """The sealed module attribute that holds `prim`, or Uncacheable."""
    held = _HOLDERS.identity(prim)
    if held is None:
      raise Uncacheable(
          f"the primitive {prim.name} is held by no module of the sealed"
          " installation")
    return seq(b"Ph", [string(held[0]), string(held[1])])

  def _primitive(self, prim: jax_core.Primitive, depth: int) -> bytes:
    is_call = prim in eval_jaxpr_lib.call_primitives
    self.primitives[prim] = is_call
    return seq(b"Pr", [
        string(prim.name), self._e(bool(prim.multiple_results), depth + 1),
        self._e(is_call, depth + 1), self._class(type(prim), "primitive class"),
        self._holder(prim),
    ])

  # -- avals, shardings, meshes

  def _aval(self, aval: Any, depth: int) -> bytes:
    return self._e(aval, depth)

  def _shaped_array(self, aval: jax_core.ShapedArray, depth: int) -> bytes:
    return seq(b"Av", [
        self._e(tuple(aval.shape), depth + 1), self._e(aval.dtype, depth + 1),
        self._e(bool(aval.weak_type), depth + 1),
        self._e(aval.sharding, depth + 1),
        self._e(aval.manual_axis_type, depth + 1),
        self._e(aval.memory_space, depth + 1), self._e(aval.layout, depth + 1),
    ])

  def _abstract_ref(self, aval: state_types.AbstractRef, depth: int,
                    *extra: Any) -> bytes:
    return seq(b"Ar", [
        self._class(type(aval), "aval class"),
        self._e(aval.inner_aval, depth + 1),
        self._e(aval.memory_space, depth + 1), self._e(aval.kind, depth + 1),
        *(self._e(e, depth + 1) for e in extra),
    ])

  def _sharding(self, x: named_sharding.NamedSharding, depth: int) -> bytes:
    return seq(b"Sh", [
        self._e(x.mesh, depth + 1), self._e(x.spec, depth + 1),
        self._e(x._memory_kind, depth + 1),
        self._e(x._logical_device_ids, depth + 1),
    ])

  def _mesh(self, mesh: Any, depth: int) -> bytes:
    if isinstance(mesh, mesh_lib.AbstractMesh):
      return seq(b"Ma", [
          self._e(tuple(mesh.axis_sizes), depth + 1),
          self._e(tuple(mesh.axis_names), depth + 1),
          self._e(tuple(mesh.axis_types), depth + 1),
          self._e(mesh.abstract_device, depth + 1),
      ])
    if mesh.empty:
      return b"Me"
    return seq(b"Mc", [
        self._e(tuple(mesh.axis_names), depth + 1),
        self._e(tuple(int(s) for s in mesh.devices.shape), depth + 1),
        self._e(tuple(int(d.id) for d in mesh.devices.flat), depth + 1),
        self._e(tuple(mesh.axis_types), depth + 1),
    ])


def _encode_shaped_array(enc: Encoder, aval: Any, depth: int) -> bytes:
  return enc._shaped_array(aval, depth)


def _encode_abstract_ref(enc: Encoder, aval: Any, depth: int) -> bytes:
  return enc._abstract_ref(aval, depth)


def _encode_tmem_ref(enc: Encoder, aval: Any, depth: int) -> bytes:
  return enc._abstract_ref(aval, depth, aval.layout, aval.collective)


def _encode_ref_union(enc: Encoder, aval: Any, depth: int) -> bytes:
  return enc._abstract_ref(aval, depth, tuple(aval.refs))


# Each aval class a lowering can see, by exact class. Any other is uncacheable.
_AVAL_ENCODERS: dict[type, Callable[[Encoder, Any, int], bytes]] = {
    jax_core.ShapedArray: _encode_shaped_array,
    state_types.AbstractRef: _encode_abstract_ref,
    gpu_core.WGMMAAbstractAccumulatorRef: _encode_abstract_ref,
    gpu_core.AbstractTMEMRef: _encode_tmem_ref,
    gpu_core.AbstractRefUnion: _encode_ref_union,
}
