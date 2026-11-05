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

"""What a Mosaic GPU lowering dispatches through, and the identity of it all.

A lowering is the rules it finds in registries: which Mosaic GPU rule lowers a
primitive (per lowering semantics), which resource estimator it has, which
primitives are calls lowered as their bodies, which state primitives are
discharged how, which MLIR op the dialect lowering lowers how and for which warp
semantics, how layouts are inferred, and the small tables those rules index (the
dialect's comparisons, its memory spaces). It also re-traces, through JAX's own
tables (staging, forwarding and folding rules, the avals of types, the MLIR types
of dtypes). The identity of all of it is computed from every entry, by the code
of the rule it holds and the state that code closes over, so a rule that is not
the installed one, one changed in place, or one that is not there, is another key.

Two kinds of table are told apart:

* *The lowering's own* (strict): every entry must be sealed code, or the process
  is unsealed. An entry keyed by a primitive that no sealed module holds is not
  one: no cacheable kernel can have that primitive (see ``Encoder``).
* *JAX's tracing tables* (by sealed key): the part that sealed code can reach,
  which is the entries keyed by something sealed (a primitive, a class, a dtype).
  An entry a library registers for its own primitive or type is none of them; a
  sealed key whose rule is not sealed code leaves the process unsealed.

The identity is computed from the entries again when they are not the objects
they were (compared at every lowering, by identity: cheap), and a lowering is
cached, and a hit returned, only if what it dispatched through is what it was
keyed under (``unchanged``): the entries are the objects they were, nothing
counted (``jax._src.counted_registry``) changed in between (a change made and
undone included), and nothing the universe holds changed
(``lowering_cache_universe``). Every table is a counted one, and a partial in
one is a frozen one (its keywords cannot be changed in place).
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from typing import Any

from jax._src import core as jax_core
from jax._src import counted_registry
from jax._src import dtypes
from jax._src import effects as jax_effects
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import pxla
from jax._src.lax import eval_jaxpr as eval_jaxpr_lib
from jax._src.lax import lax as lax_internal
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.pallas.mosaic_gpu import lowering_cache_encoding as encoding
from jax._src.pallas.mosaic_gpu import lowering_cache_universe as universe
from jax._src.pallas.mosaic_gpu import lowering_cache_seal as sealing
from jax._src.state import discharge
from jax._src.state import indexing
from jax.experimental.mosaic.gpu import dialect_lowering
from jax.experimental.mosaic.gpu import layout_inference
from jax.experimental.mosaic.gpu import utils as mgpu_utils


@dataclasses.dataclass(frozen=True)
class Table:
  name: str
  # Resolved when the identity is computed: the object a module holds then.
  get: Any
  # Whether every entry must be sealed (else, only those of a sealed key).
  strict: bool


def _lowering_tables() -> list[Table]:
  tables = [
      Table(f"mosaic_lowering_rules[{semantics}]",
            lambda s=semantics: lowering.mosaic_lowering_rules[s], True)
      for semantics in sorted(lowering.mosaic_lowering_rules, key=repr)
  ]
  tables += [
      Table("mosaic_gpu.lowering._resource_estimators",
            lambda: lowering._resource_estimators, True),
      Table("jax.lax.eval_jaxpr.call_primitives",
            lambda: eval_jaxpr_lib.call_primitives, True),
      Table("jax.state.discharge._discharge_rules",
            lambda: discharge._discharge_rules, True),
      Table("mosaic_gpu.dialect_lowering._lowerings",
            lambda: dialect_lowering._lowerings, True),
      Table("mosaic_gpu.dialect_lowering._supported_warp_lowerings",
            lambda: dialect_lowering._supported_warp_lowerings, True),
      Table("mosaic_gpu.layout_inference._constraint_system_derivation_rules",
            lambda: layout_inference._constraint_system_derivation_rules, True),
      Table("mosaic_gpu.dialect_lowering.CMPI_IMPLS",
            lambda: dialect_lowering.CMPI_IMPLS, True),
      Table("mosaic_gpu.dialect_lowering.CMPF_IMPLS",
            lambda: dialect_lowering.CMPF_IMPLS, True),
      Table("mosaic_gpu.utils._MEMORY_SPACES",
            lambda: mgpu_utils._MEMORY_SPACES, True),
  ]
  return tables


# The tables of JAX's tracing that a lowering reads when it re-traces a jaxpr
# (the discharge of state, the lowering of control flow through `lax`). Named by
# the module that defines each, as the audit of what a lowering reads names them.
_TRACING_TABLES = (
    ("jax._src.interpreters.partial_eval.custom_staging_rules",
     lambda: pe.custom_staging_rules),
    ("jax._src.interpreters.partial_eval.forwarding_rules",
     lambda: pe.forwarding_rules),
    ("jax._src.interpreters.partial_eval.const_fold_rules",
     lambda: pe.const_fold_rules),
    ("jax._src.core.pytype_aval_mappings", lambda: jax_core.pytype_aval_mappings),
    ("jax._src.core.literalable_types", lambda: jax_core.literalable_types),
    ("jax._src.core.literalable_scalar_types",
     lambda: jax_core.literalable_scalar_types),
    ("jax._src.interpreters.mlir._dtype_to_ir_type",
     lambda: mlir._dtype_to_ir_type),
    ("jax._src.dtypes._DEFAULT_TYPEMAP", lambda: dtypes._DEFAULT_TYPEMAP),
    ("jax._src.dtypes._dtype_to_32bit_dtype",
     lambda: dtypes._dtype_to_32bit_dtype),
    ("jax._src.dtypes.python_scalar_types_to_dtypes",
     lambda: dtypes.python_scalar_types_to_dtypes),
    ("jax._src.dtypes._custom_float_dtypes", lambda: dtypes._custom_float_dtypes),
    ("jax._src.dtypes._custom_float_scalar_types",
     lambda: dtypes._custom_float_scalar_types),
    ("jax._src.dtypes._jax_dtype_set", lambda: dtypes._jax_dtype_set),
    ("jax._src.dtypes._registered_weak_types",
     lambda: dtypes._registered_weak_types),
    ("jax._src.dtypes._weak_types", lambda: dtypes._weak_types),
    ("jax._src.dtypes.python_scalar_types", lambda: dtypes.python_scalar_types),
    ("jax._src.interpreters.pxla.shard_arg_handlers",
     lambda: pxla.shard_arg_handlers),
    ("jax._src.effects.control_flow_allowed_effects",
     lambda: jax_effects.control_flow_allowed_effects._effect_types),
    ("jax._src.lax.lax._foldable_types", lambda: lax_internal._foldable_types),
    ("jax._src.state.discharge._neutral_memory_spaces",
     lambda: discharge._neutral_memory_spaces),
    ("jax._src.state.indexing.indexer_transform_type_registry",
     lambda: indexing.indexer_transform_type_registry),
)


def tables() -> list[Table]:
  """Every table the identity is made of."""
  return _lowering_tables() + [
      Table(name, get, False) for name, get in _TRACING_TABLES
  ]


def _encode_table(enc: encoding.Encoder, table: Table) -> bytes:
  """A table's entries: each key and the code (and state) of what it holds."""
  live = table.get()
  if isinstance(live, (set, frozenset, list)):
    entries = [(element, None) for element in list(live)]
  else:
    entries = list(live.items())
  encoded = []
  for key, rule in entries:
    try:
      key_encoding = enc.encode(key)
    except encoding.Uncacheable as e:
      # A primitive no sealed module holds is not one a cacheable kernel has, and
      # for the tracing tables what no sealed code can reach is no entry.
      if not table.strict or isinstance(key, jax_core.Primitive):
        continue
      raise sealing.Unsealed(f"{table.name}[{key}]: {e}") from e
    try:
      rule_encoding = b"" if rule is None else enc.encode(rule)
    except encoding.Uncacheable as e:
      raise sealing.Unsealed(
          f"{table.name}[{getattr(key, 'name', key)}]: {e}") from e
    encoded.append(key_encoding + rule_encoding)
  return encoding.seq(b"table", [encoding.string(table.name), *sorted(encoded)])


def _tables_token() -> tuple[Any, ...]:
  """What the tables are made of, by identity: cheap, and exact for what is immutable."""
  token = []
  for table in tables():
    live = table.get()
    if isinstance(live, (set, frozenset)):
      token.append((table.name, id(live), frozenset(map(id, live))))
    elif isinstance(live, list):
      token.append((table.name, id(live), tuple(map(id, live))))
    else:
      token.append((table.name, id(live), tuple(
          (id(key), universe.callable_token(rule)) for key, rule in live.items())))
  return tuple(token)


def _require_counted() -> None:
  """Raises Unsealed if a table is not the counted one its module defines."""
  for table in tables():
    live = table.get()
    if not isinstance(
        live, (counted_registry.CountedDict, counted_registry.CountedSet,
               counted_registry.CountedList)) or (
            counted_registry.named(live.name) is not live):
      raise sealing.Unsealed(f"{table.name} is not the table its module defines")


_cache_lock = threading.RLock()
_last_tables: tuple[tuple[Any, ...], bytes] | None = None


def _tables_digest(seal: sealing.Seal) -> tuple[tuple[Any, ...], bytes]:
  global _last_tables
  with _cache_lock:
    token = _tables_token()
    if _last_tables is not None and _last_tables[0] == token:
      return _last_tables
    enc = encoding.Encoder()
    digest = hashlib.sha256(
        b"".join(_encode_table(enc, table) for table in tables())).digest()
    _last_tables = (token, digest)
    return _last_tables


@dataclasses.dataclass(frozen=True)
class State:
  """What the lowering dispatches through: the count of what changed at the time
  (`counted_registry.epoch`), what the tables were made of, and the identity of
  all of it (`current`)."""
  epoch: int
  tables: tuple[Any, ...]
  digest: bytes


def current() -> State:
  """The identity of what a lowering dispatches through, as it is now.

  Raises Unsealed if an entry is not sealed code, a function of the universe is
  not what was sealed, or what it is made of changes while this reads it.
  """
  seal = sealing.seal()
  # Registers the rules of the primitives, which a kernel that uses one would
  # otherwise find only once it imports them.
  from jax._src.pallas.mosaic_gpu import primitives  # pylint: disable=unused-import,g-import-not-at-top
  del primitives
  _require_counted()
  for _ in range(3):
    epoch = counted_registry.epoch()
    universe_digest = universe.check(seal)
    tables, tables_digest = _tables_digest(seal)
    if counted_registry.epoch() == epoch:
      return State(epoch, tables, hashlib.sha256(
          universe_digest + tables_digest).digest())
  raise sealing.Unsealed(
      "what a lowering dispatches through changed while it was read")


def unchanged(since: State, *, deep: bool = True) -> bool:
  """Whether what a lowering dispatches through is what `since` was.

  Nothing counted changed (not even to change back), the tables are made of the
  objects they were, and (if `deep`) the universe holds what it did, which takes
  a look at every function and class of it. A hit, which is read and rebuilt in a
  moment, is not looked at as deeply as a lowering of seconds.
  """
  return (
      counted_registry.epoch() == since.epoch
      and _tables_token() == since.tables
      and (not deep or universe.tokens_unchanged())
  )


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
