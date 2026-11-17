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

"""The constants a Mosaic GPU lowering reads, which the key holds by value.

In a process that has not changed JAX a constant is what the installed code
defines, which the seal hashes. A process that assigns one a value of its own
(``gpu_core.SMEM_ALIGNMENT = 2048``) has changed what the lowering does, and a
key made of the installation alone would not tell it from the stock one.

Which names are constants is read from the sealed source of the lowering's
packages, not from what the modules hold now (a constant replaced before the
first look would otherwise not be one): a name is a constant if the module's
source assigns it a literal at module level, and never ``global``-assigns it in a
function (a counter is no constant). The value each holds is then read at every
lowering, and is part of the key through the strict encoder. A name that is gone,
or holds anything but the literal kind of value its source assigns it (a number,
a string, ``None``, a tuple of them; not an object, a NumPy scalar or an instance
of a subclass of ``int``), leaves the process unsealed.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
import threading
import types
from typing import Any

from jax._src.pallas.mosaic_gpu import lowering_cache_encoding as encoding
from jax._src.pallas.mosaic_gpu import lowering_cache_seal as sealing

# The packages whose code a Mosaic GPU lowering runs, by module name.
_SCOPE = ("jax._src.pallas.", "jax.experimental.mosaic.gpu")
# The cache's own modules, which the lowering does not run.
_EXCLUDED = (
    "jax._src.pallas.mosaic_gpu.lowering_cache",
    "jax.experimental.mosaic.gpu.lowering_inputs",
)


_GLOBAL = re.compile(r"^[ \t]*global[ \t]+([\w, \t]+)", re.MULTILINE)


def in_scope(name: str) -> bool:
  return name.startswith(_SCOPE) and not name.startswith(_EXCLUDED)


def _is_literal(node: ast.AST) -> bool:
  if isinstance(node, ast.Constant):
    return node.value is not Ellipsis
  if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
    return _is_literal(node.operand)
  if isinstance(node, ast.Tuple):
    return all(_is_literal(e) for e in node.elts)
  return False


def _statements(body: list[ast.stmt]):
  """The statements of a module, those under `if`, `try` and `with` included."""
  for node in body:
    yield node
    if isinstance(node, ast.If):
      yield from _statements(node.body)
      yield from _statements(node.orelse)
    elif isinstance(node, ast.Try):
      for block in (node.body, node.orelse, node.finalbody,
                    *(h.body for h in node.handlers)):
        yield from _statements(block)
    elif isinstance(node, ast.With):
      yield from _statements(node.body)


def _stored_names(node: ast.AST) -> set[str]:
  return {
      n.id for n in ast.walk(node)
      if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
  }


def constant_names(source: bytes, filename: str) -> tuple[str, ...]:
  """The names a module's source assigns a literal at module level, and no more.

  A name that the module also defines, imports, deletes or assigns anything but a
  literal (in a loop, a handler, a tuple target, an augmented assignment), or that
  a function assigns through ``global``, is not one: its value is no more than
  what the installed code computes, or is the state of the process.
  """
  tree = ast.parse(source, filename)
  literal: set[str] = set()
  other: set[str] = set()
  for node in _statements(tree.body):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
      for target in targets:
        if isinstance(target, ast.Name):
          is_literal = node.value is not None and _is_literal(node.value)
          (literal if is_literal else other).add(target.id)
        else:  # A tuple, an attribute, a subscript: not a single name's value.
          other.update(_stored_names(target))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      other.add(node.name)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
      other.update(
          (a.asname or a.name).partition(".")[0] for a in node.names)
    elif isinstance(node, (ast.If, ast.Try, ast.With)):
      # Their bodies are visited as statements; what they bind themselves.
      if isinstance(node, ast.With):
        for item in node.items:
          if item.optional_vars is not None:
            other.update(_stored_names(item.optional_vars))
      if isinstance(node, ast.Try):
        other.update(h.name for h in node.handlers if h.name)
    else:
      other.update(_stored_names(node))
  # A name a function assigns through `global` is state of the process. (Found
  # by its line, which is cheaper than walking every function; one that a string
  # or a comment mentions is only left out, which a constant can be.)
  declared = {
      name.strip()
      for match in _GLOBAL.finditer(source.decode("utf-8", "replace"))
      for name in match.group(1).split(",")
  }
  return tuple(sorted(
      name for name in literal - other - declared
      if not (name.startswith("__") and name.endswith("__"))
  ))


def _is_literal_value(value: Any) -> bool:
  """Whether a value is of the literal kind a source assigns a constant."""
  t = type(value)
  if t is tuple:
    return all(_is_literal_value(v) for v in value)
  return value is None or t in (bool, int, float, complex, str, bytes)


_MISSING = object()


class _Inventory:
  """The constants of the loaded modules in scope, and the digest of their values."""

  def __init__(self):
    self._lock = threading.Lock()
    # module name -> (the module, the names its source says are constants).
    self._modules: dict[str, tuple[types.ModuleType, tuple[str, ...]]] = {}
    self._seen = -1  # len(sys.modules) when the modules were last looked for.
    self._token: tuple[int, ...] | None = None
    self._values: list[Any] = []
    self._digest = b""

  def _scan(self, seal: sealing.Seal) -> bool:
    """Reads the names of the modules loaded since the last look. True if any."""
    added = False
    for name, module in list(sys.modules.items()):
      if name in self._modules or module is None or not in_scope(name):
        continue
      file = getattr(module, "__file__", None)
      if not file or not file.endswith(".py") or not seal.holds(file):
        continue  # (A module no RECORD lists is `Seal.verify_loaded`'s.)
      try:
        with open(file, "rb") as f:
          names = constant_names(f.read(), file)
      except (OSError, SyntaxError, ValueError) as e:
        raise sealing.Unsealed(f"the source of {name} cannot be read: {e}") from e
      self._modules[name] = (module, names)
      added = True
    self._seen = len(sys.modules)
    return added

  def digest(self, seal: sealing.Seal) -> bytes:
    """The digest of the values the constants hold now.

    Raises Unsealed if one is gone or holds a value the key cannot hold exactly.
    """
    with self._lock:
      changed = False
      if len(sys.modules) != self._seen:
        changed = self._scan(seal)
      values = [
          vars(module).get(name, _MISSING)
          for module, names in self._modules.values() for name in names
      ]
      token = tuple(map(id, values))
      if not changed and token == self._token:
        return self._digest
      encoder = encoding.Encoder()
      parts = []
      index = 0
      for module_name, (_, names) in self._modules.items():
        for name in names:
          value = values[index]
          index += 1
          if value is _MISSING:
            raise sealing.Unsealed(
                f"{module_name}.{name}, a constant of the installed code, is gone")
          if not _is_literal_value(value):
            raise sealing.Unsealed(
                f"{module_name}.{name} is a constant of the installed code and"
                f" holds a {type(value).__qualname__}, which is not the literal"
                " kind of value the source assigns it")
          encoded = encoder.encode(value)
          parts.append(encoding.string(f"{module_name}.{name}") + encoded)
      # `values` is kept: no object of the token is freed and its id reused.
      self._values = values
      self._token = token
      self._digest = hashlib.sha256(b"".join(sorted(parts))).digest()
      return self._digest


_inventory = _Inventory()


def digest() -> bytes:
  """The digest of the values of the lowering's constants (see the module's doc)."""
  return _inventory.digest(sealing.seal())
