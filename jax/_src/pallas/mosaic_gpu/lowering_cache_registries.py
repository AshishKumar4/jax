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

"""What a Mosaic GPU lowering dispatches through: the rule registries.

A lowering is the rules it finds in these registries: which Mosaic GPU rule
lowers a primitive (per lowering semantics), which resource estimator it has,
which primitives are calls lowered as their bodies, which state primitives are
discharged how, which MLIR op the dialect lowering lowers how and for which warp
semantics, and how layouts are inferred. Registering a rule is how a library
extends a lowering, so the registries are part of what a key names: each entry,
by its key and by the code of the rule it holds, is hashed into ``State.digest``.

Each rule is a function, a method or a partial of sealed code (see
``Encoder``), or the process is unsealed. An entry keyed by a primitive that no
sealed module holds is no entry of a kernel that can be cached (such a primitive
is uncacheable, see ``Encoder``), and is left out.

The registries are read by identity: ``State.token`` is the objects each holds,
and the digest is made again only when they are not the objects it was made of.
``unchanged`` tells a caller whether they still are, which is how a rule that
registers another while a kernel lowers, or a registry that is replaced after
the first lowering, is found.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from typing import Any

from jax._src import core as jax_core
from jax._src.lax import eval_jaxpr as eval_jaxpr_lib
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.pallas.mosaic_gpu import lowering_cache_encoding as encoding
from jax._src.pallas.mosaic_gpu import lowering_cache_seal as sealing
from jax._src.state import discharge
from jax.experimental.mosaic.gpu import dialect_lowering
from jax.experimental.mosaic.gpu import layout_inference


@dataclasses.dataclass(frozen=True)
class Table:
  name: str
  # Resolved when the identity is computed: the object a module holds then.
  get: Any


def tables() -> list[Table]:
  """Every registry a lowering dispatches through."""
  rules = [
      Table(f"mosaic_lowering_rules[{semantics}]",
            lambda s=semantics: lowering.mosaic_lowering_rules[s])
      for semantics in sorted(lowering.mosaic_lowering_rules, key=repr)
  ]
  return rules + [
      Table("mosaic_gpu.lowering._resource_estimators",
            lambda: lowering._resource_estimators),
      Table("jax.lax.eval_jaxpr.call_primitives",
            lambda: eval_jaxpr_lib.call_primitives),
      Table("jax.state.discharge._discharge_rules",
            lambda: discharge._discharge_rules),
      Table("mosaic_gpu.dialect_lowering._lowerings",
            lambda: dialect_lowering._lowerings),
      Table("mosaic_gpu.dialect_lowering._supported_warp_lowerings",
            lambda: dialect_lowering._supported_warp_lowerings),
      Table("mosaic_gpu.layout_inference._constraint_system_derivation_rules",
            lambda: layout_inference._constraint_system_derivation_rules),
  ]


def _entries(live: Any) -> list[tuple[Any, Any]]:
  if isinstance(live, (set, frozenset)):
    return [(element, None) for element in list(live)]
  return list(live.items())


def _encode_table(enc: encoding.Encoder, table: Table) -> bytes:
  """A table's entries: each key and the code (and state) of what it holds."""
  encoded = []
  for key, rule in _entries(table.get()):
    try:
      key_encoding = enc.encode(key)
    except encoding.Uncacheable as e:
      # A primitive no sealed module holds is not one a cacheable kernel has.
      if isinstance(key, jax_core.Primitive):
        continue
      raise sealing.Unsealed(f"{table.name}[{key}]: {e}") from e
    try:
      rule_encoding = b"" if rule is None else enc.encode(rule)
    except encoding.Uncacheable as e:
      raise sealing.Unsealed(
          f"{table.name}[{getattr(key, 'name', key)}]: {e}") from e
    encoded.append(key_encoding + rule_encoding)
  return encoding.seq(b"table", [encoding.string(table.name), *sorted(encoded)])


def _token() -> tuple[Any, ...]:
  """What the registries are made of, by identity: cheap to make and compare."""
  return tuple(
      (table.name, id(live := table.get()),
       tuple((id(key), id(rule)) for key, rule in _entries(live)))
      for table in tables()
  )


@dataclasses.dataclass(frozen=True)
class State:
  """What the registries held, and the identity of it all."""
  token: tuple[Any, ...]
  digest: bytes
  # Held so that no object of the token is freed (and its id reused) meanwhile.
  entries: tuple[Any, ...] = dataclasses.field(repr=False, compare=False)


_lock = threading.Lock()
_last: State | None = None


def current() -> State:
  """The identity of what a lowering dispatches through, as it is now.

  Raises Unsealed if an entry is not sealed code.
  """
  global _last
  # Registers the rules of the primitives, which a kernel that uses one would
  # otherwise find only once it imports them.
  from jax._src.pallas.mosaic_gpu import primitives  # pylint: disable=unused-import,g-import-not-at-top
  del primitives
  with _lock:
    token = _token()
    if _last is not None and _last.token == token:
      return _last
    enc = encoding.Encoder()
    digest = hashlib.sha256(
        b"".join(_encode_table(enc, table) for table in tables())).digest()
    entries = tuple(
        (live := table.get(), tuple(_entries(live))) for table in tables())
    _last = State(token, digest, entries)
    return _last


def unchanged(since: State) -> bool:
  """Whether the registries are the objects `since` was made of."""
  return _token() == since.token


def check_primitives(primitives: dict[jax_core.Primitive, bool]) -> None:
  """Raises Unsealed unless each primitive of a kernel has a rule it is keyed by.

  A primitive is a call primitive (lowered as its body, which the key holds), or
  has a Mosaic GPU lowering rule or a state discharge rule, as the very object
  the registry holds (a primitive of the same name that is another object has
  none).
  """
  known: set[jax_core.Primitive] = set(discharge._discharge_rules)
  for rules in list(lowering.mosaic_lowering_rules.values()):
    known.update(rules)
  for primitive, is_call in primitives.items():
    if not is_call and primitive not in known:
      raise sealing.Unsealed(
          f"no sealed lowering or discharge rule for the primitive {primitive.name}"
      )
