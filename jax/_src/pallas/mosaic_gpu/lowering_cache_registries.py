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
extends a lowering, so the rules a lowering can reach are part of what its key
names: each entry, by its key and by the code of the rule it holds.

What a lowering reaches. A kernel's equations are lowered by their primitives'
Mosaic GPU rules, and so are the equations its rules make as they lower (a rule
lowering a function's trace, a body's discharged form, a call lowered as
``eval_jaxpr_p``): each holds a primitive a Mosaic GPU rule lowers, or a call
primitive. A state discharge rule runs on a body one of the kernel's equations
holds (a loop's, ``run_state``'s), so on the kernel's own primitives, or on one
a rule made, whose ref operations are state primitives. The dialect's tables
are reached by whatever ops a module holds. So every lowering reaches the
entries, in every table, of each primitive that has a Mosaic GPU rule, is a
call primitive or is a state primitive, and the dialect's tables whole
(``State.digest``);
a kernel reaches as well the entries of its own primitives that are neither
(``State.reached``). An entry of any other primitive is in no key: a module
imported late that registers a rule for a primitive no kernel holds (a state
discharge rule for one of its own) changes no key, and one whose rule is not
sealed code bypasses no lowering.

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
from collections.abc import Iterable
from typing import Any

from jax._src import core as jax_core
from jax._src.lax import eval_jaxpr as eval_jaxpr_lib
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.pallas.mosaic_gpu import lowering_cache_encoding as encoding
from jax._src.pallas.mosaic_gpu import lowering_cache_seal as sealing
from jax._src.state import discharge
from jax._src.state import primitives as state_primitives
from jax.experimental.mosaic.gpu import dialect_lowering
from jax.experimental.mosaic.gpu import layout_inference


@dataclasses.dataclass(frozen=True)
class Table:
  name: str
  # Resolved when the identity is computed: the object a module holds then.
  get: Any
  # Whether it is keyed by primitives, of which a key holds the entries a
  # lowering reaches; else every entry is reached.
  by_primitive: bool


def tables() -> list[Table]:
  """Every registry a lowering dispatches through."""
  rules = [
      Table(f"mosaic_lowering_rules[{semantics}]",
            lambda s=semantics: lowering.mosaic_lowering_rules[s], True)
      for semantics in sorted(lowering.mosaic_lowering_rules, key=repr)
  ]
  return rules + [
      Table("mosaic_gpu.lowering._resource_estimators",
            lambda: lowering._resource_estimators, True),
      Table("jax.lax.eval_jaxpr.call_primitives",
            lambda: eval_jaxpr_lib.call_primitives, True),
      Table("jax.state.discharge._discharge_rules",
            lambda: discharge._discharge_rules, True),
      Table("mosaic_gpu.dialect_lowering._lowerings",
            lambda: dialect_lowering._lowerings, False),
      Table("mosaic_gpu.dialect_lowering._supported_warp_lowerings",
            lambda: dialect_lowering._supported_warp_lowerings, False),
      Table("mosaic_gpu.layout_inference._constraint_system_derivation_rules",
            lambda: layout_inference._constraint_system_derivation_rules,
            False),
  ]


def _entries(live: Any) -> list[tuple[Any, Any]]:
  if isinstance(live, (set, frozenset)):
    return [(element, None) for element in list(live)]
  return list(live.items())


def _encode_entry(
    enc: encoding.Encoder, table: Table, key: Any, rule: Any
) -> bytes | None:
  """An entry: its key and the code (and state) of what it holds; None for
  one keyed by a primitive no sealed module holds."""
  try:
    key_encoding = enc.encode(key)
  except encoding.Uncacheable as e:
    # A primitive no sealed module holds is not one a cacheable kernel has.
    if isinstance(key, jax_core.Primitive):
      return None
    raise sealing.Unsealed(f"{table.name}[{key}]: {e}") from e
  try:
    rule_encoding = b"" if rule is None else enc.encode(rule)
  except encoding.Uncacheable as e:
    raise sealing.Unsealed(
        f"{table.name}[{getattr(key, 'name', key)}]: {e}") from e
  return key_encoding + rule_encoding


def _token() -> tuple[Any, ...]:
  """What the registries are made of, by identity: cheap to make and compare."""
  return tuple(
      (table.name, id(live := table.get()),
       tuple((id(key), id(rule)) for key, rule in _entries(live)))
      for table in tables()
  )


@dataclasses.dataclass(frozen=True)
class State:
  """What the registries held, and the identity of what every lowering
  reaches of it (`digest`: the entries of the primitives in `lowered` and the
  dialect's tables); and whether any rule of that runs an extension's
  code, a declared source package's, however a sealed rule holds it
  (`extended`, as their encoding met it), which a lowering may hand anything it
  holds, its mesh among it."""
  token: tuple[Any, ...]
  digest: bytes
  # Held so that no object of the token is freed (and its id reused) meanwhile.
  entries: tuple[Any, ...] = dataclasses.field(repr=False, compare=False)
  extended: bool = False
  # The primitives every lowering reaches the entries of (`current`'s).
  lowered: frozenset[jax_core.Primitive] = dataclasses.field(
      default=frozenset(), repr=False, compare=False)
  # Each primitive-keyed table and its entries, as `entries` holds them.
  primitive_tables: tuple[tuple[Table, dict[Any, Any]], ...] = (
      dataclasses.field(default=(), repr=False, compare=False))
  # Each primitive outside `lowered` a kernel held: its entries' encoding and
  # whether it met an extension's code (`reached`), made once.
  _own: dict[jax_core.Primitive, tuple[bytes, bool]] = dataclasses.field(
      default_factory=dict, repr=False, compare=False)

  def reached(
      self, primitives: Iterable[jax_core.Primitive]
  ) -> tuple[bytes, bool]:
    """What lowering a kernel of `primitives` (its encoding's,
    `Encoder.primitives`) reaches: the identity of `digest` and of the entries
    of those primitives outside `lowered` (a state discharge rule, run on a body
    an equation holds), and whether any rule of it runs an extension's code.

    Raises Unsealed if an entry it reaches is not sealed code.
    """
    own = []
    extended = self.extended
    for primitive in primitives:
      if primitive in self.lowered:
        continue
      found = self._own.get(primitive)
      if found is None:
        enc = encoding.Encoder()
        held = []
        for table, live in self.primitive_tables:
          if primitive in live and (entry := _encode_entry(
              enc, table, primitive, live[primitive])) is not None:
            held.append(
                encoding.seq(b"entry", [encoding.string(table.name), entry]))
        # Lowerings on several threads may each make it: the same bytes.
        found = self._own.setdefault(primitive, (b"".join(held), enc.extended))
      if found[0]:
        own.append(found[0])
        extended |= found[1]
    if not own:
      return self.digest, extended
    return hashlib.sha256(self.digest + b"".join(sorted(own))).digest(), extended


def _lowered() -> set[jax_core.Primitive]:
  """The primitives every lowering reaches the entries of: each a Mosaic GPU
  rule lowers or a call primitive, which is what a rule's equations hold; and
  the state primitives, the only ones holding a ref that JAX's own code makes,
  so the only ones a body a discharge rule makes could hold for a later
  discharge."""
  # Each table copied before it is read: a rule may be registered meanwhile.
  lowered: set[jax_core.Primitive] = set(eval_jaxpr_lib.call_primitives.copy())
  for rules in list(lowering.mosaic_lowering_rules.values()):
    lowered.update(rules.copy())
  lowered.update(value for value in vars(state_primitives).values()
                 if isinstance(value, jax_core.Primitive))
  return lowered


_lock = threading.Lock()
_last: State | None = None


def current() -> State:
  """The identity of what a lowering dispatches through, as it is now.

  Raises Unsealed if an entry every lowering reaches is not sealed code.
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
    snapshot = [(table, (live := table.get()), tuple(_entries(live)))
                for table in tables()]
    lowered = _lowered()
    enc = encoding.Encoder()
    encoded = []
    for table, _, held in snapshot:
      reached = (entry for entry in held
                 if not table.by_primitive or entry[0] in lowered)
      parts = [part for key, rule in reached
               if (part := _encode_entry(enc, table, key, rule)) is not None]
      encoded.append(encoding.seq(
          b"table", [encoding.string(table.name), *sorted(parts)]))
    digest = hashlib.sha256(b"".join(encoded)).digest()
    # Whatever code the rules run, as their encoding met it (`Encoder.extended`).
    _last = State(
        token, digest, tuple((live, held) for _, live, held in snapshot),
        enc.extended, frozenset(lowered),
        tuple((table, dict(held)) for table, _, held in snapshot
              if table.by_primitive))
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
  known: set[jax_core.Primitive] = set(discharge._discharge_rules.copy())
  for rules in list(lowering.mosaic_lowering_rules.values()):
    known.update(rules.copy())
  for primitive, is_call in primitives.items():
    if not is_call and primitive not in known:
      raise sealing.Unsealed(
          f"no sealed lowering or discharge rule for the primitive {primitive.name}"
      )
