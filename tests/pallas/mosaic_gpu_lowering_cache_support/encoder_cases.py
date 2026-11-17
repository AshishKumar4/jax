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
"""The canonicalizer's cases: two values, and whether their encodings differ.

Each case is (name, a, b, expected), where expected is "equal" (one encoding),
"different" or "uncacheable" (the encoder raises `Uncacheable` for either). They
run in a process that has the sealed site's jax, as sealed classes (enums,
dataclasses, avals) are only encodable where the installation is sealed.
"""

import dataclasses
import enum
import struct

from jax import lax
import jax.numpy as jnp
import jax.tree_util as tree_util
import numpy as np

from jax._src import core as jax_core
from jax._src import literals
from jax._src import lowering_cache_test_classes as sealed_classes
from jax._src import lowering_cache_test_rules as sealed_rules
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering_cache_encoding as cache_encoding
from jax._src.state import types as state_types
from jax.experimental.mosaic.gpu import tcgen05


@dataclasses.dataclass(frozen=True)
class Outside:
  value: int


class OutsideEnum(enum.Enum):
  A = 1


@tree_util.register_pytree_node_class
class OutsideNode:

  def __init__(self, x):
    self.x = x

  def tree_flatten(self):
    return (self.x,), None

  @classmethod
  def tree_unflatten(cls, aux, children):
    return cls(*children)


class Local:
  pass


def outside_function(f):
  def wrapper(*args, **kwargs):
    return f(*args, **kwargs)
  return wrapper


# A primitive that is named "add" as `lax.add_p` is, and that no sealed module holds.
_unsealed_add_p = jax_core.Primitive("add")


def _bits(b):
  return struct.unpack("<d", struct.pack("<Q", b))[0]


def cases():
  o1, o2, o3, o4 = object(), object(), object(), object()
  f32 = np.dtype(np.float32)
  ref = lambda space: state_types.AbstractRef(
      jax_core.ShapedArray((8,), f32), space)
  layout = tcgen05.tmem_default_layout(packing=1)
  tmem = lambda collective: gpu_core.AbstractTMEMRef(
      jax_core.ShapedArray((128, 8), f32), gpu_core.TMEM, layout, collective)
  return [
      ("-0.0 and 0.0", 0.0, -0.0, "different"),
      ("equal floats", 1.5, 1.5, "equal"),
      ("NaN payloads", _bits(0x7ff8000000000000), _bits(0x7ff8000000000001), "different"),
      ("the same NaN", _bits(0x7ff8000000000001), _bits(0x7ff8000000000001), "equal"),
      ("True and 1", True, 1, "different"),
      ("1 and 1.0", 1, 1.0, "different"),
      ("int and np.int32", 1, np.int32(1), "different"),
      ("None and False", None, False, "different"),
      ("tuple and list", (1, 2), [1, 2], "different"),
      ("str and bytes", "a", b"a", "different"),
      ("dict order", {"a": 1, "b": 2}, {"b": 2, "a": 1}, "equal"),
      ("dict values", {"a": 1}, {"a": 2}, "different"),
      ("frozenset order", frozenset({1, 2, 3}), frozenset({3, 2, 1}), "equal"),
      ("bfloat16 and float16", np.dtype(jnp.bfloat16), np.dtype(np.float16), "different"),
      ("array dtype", np.array([1], np.int32), np.array([1.4e-45], np.float32), "different"),
      ("array values", np.array([1, 2]), np.array([1, 3]), "different"),
      ("weak and strong type", jax_core.ShapedArray((), f32, weak_type=True),
       jax_core.ShapedArray((), f32, weak_type=False), "different"),
      ("weak aval alike", jax_core.ShapedArray((), f32, weak_type=True),
       jax_core.ShapedArray((), f32, weak_type=True), "equal"),
      ("typed and plain float", literals.TypedFloat(1.0, np.dtype(np.float32)), 1.0, "different"),
      ("ref memory space", ref(gpu_core.SMEM), ref(gpu_core.GMEM), "different"),
      ("tmem ref collective", tmem(False), tmem(True), "different"),
      ("tmem ref alike", tmem(True), tmem(True), "equal"),
      ("compare=False field", sealed_classes.WithIgnored(1, 2), sealed_classes.WithIgnored(1, 3), "different"),
      ("compare=False alike", sealed_classes.WithIgnored(1, 2), sealed_classes.WithIgnored(1, 2), "equal"),
      ("sealed enum", gpu_core.MemorySpace.SMEM, gpu_core.MemorySpace.GMEM, "different"),
      ("treedef structure", tree_util.tree_structure((1, 2)), tree_util.tree_structure([1, 2]), "different"),
      ("treedef alike", tree_util.tree_structure({"a": 1}), tree_util.tree_structure({"a": 2}), "equal"),
      ("frozen dataclass outside", Outside(1), Outside(1), "uncacheable"),
      ("enum outside", OutsideEnum.A, OutsideEnum.A, "uncacheable"),
      ("pytree node outside", tree_util.tree_structure(OutsideNode(1)),
       tree_util.tree_structure(OutsideNode(1)), "uncacheable"),
      ("function outside", outside_function, outside_function, "uncacheable"),
      ("mutable dataclass", sealed_classes.Mutable(1), sealed_classes.Mutable(1), "uncacheable"),
      ("set", {1}, {1}, "uncacheable"),
      # A bare object is an axis name warp_map makes: only which other
      # occurrences it equals is anything of it.
      ("one anonymous object twice", (o1, o1), (o1, o2), "different"),
      ("anonymous objects alike", (o1, o2), (o3, o4), "equal"),
      ("an instance of a class outside", Local(), Local(), "uncacheable"),
      ("an aval class with no encoder", jax_core.abstract_token, jax_core.abstract_token, "uncacheable"),
      ("the same primitive", lax.add_p, lax.add_p, "equal"),
      ("primitives of one name that two sealed modules hold", lax.add_p,
       sealed_rules.add_a_p, "different"),
      ("a primitive no sealed module holds", _unsealed_add_p, _unsealed_add_p, "uncacheable"),
  ]


def run():
  """What each case encodes to, against what it should: {name: outcome}, {name: expected}."""
  outcomes, expected = {}, {}
  for name, a, b, want in cases():
    expected[name] = want
    try:
      encoded_a = cache_encoding.Encoder().encode(a)
      encoded_b = cache_encoding.Encoder().encode(b)
      outcomes[name] = "equal" if encoded_a == encoded_b else "different"
    except cache_encoding.Uncacheable:
      outcomes[name] = "uncacheable"
  return {"outcomes": outcomes, "expected": expected}
