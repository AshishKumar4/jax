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

"""Primitive.elementwise, Primitive.inlined_call and Primitive.transparent_call hold of what they mark,
and the primitives an analysis dispatches on are exported."""

import itertools

from absl.testing import absltest
import numpy as np

import jax
import jax.extend as jex
from jax._src import ad_util
from jax._src import core
from jax._src import test_util as jtu
from jax._src.lax import lax as lax_internal
from jax._src.lax import special
from jax._src.random import threefry2x32
import jax.custom_batching
import jax.numpy as jnp

jax.config.parse_flags_with_absl()

# The parameters a marked primitive binds with that its operands do not imply.
_PARAMS = {
    "integer_pow": {"y": 3},
    "convert_element_type": {"new_dtype": np.dtype(np.float32), "weak_type": False, "sharding": None},
    "reduce_precision": {"exponent_bits": 5, "mantissa_bits": 10},
    "round": {"rounding_method": lax_internal.RoundingMethod.TO_NEAREST_EVEN},
}

_DTYPES = (np.float32, np.int32, np.uint32, np.bool_, np.complex64)


def _marked():
  modules = (lax_internal, special, ad_util, threefry2x32)
  found = {value for module in modules for value in vars(module).values()
           if isinstance(value, core.Primitive) and value.elementwise}
  return sorted(found, key=lambda p: p.name)


def _drawn(count, dtype, rng):
  if dtype is np.bool_:
    return [rng.random((2, 3)) > 0.5 for _ in range(count)]
  if dtype is np.complex64:
    return [(rng.uniform(0.1, 0.9, (2, 3)) + 1j * rng.uniform(0.1, 0.9, (2, 3))).astype(dtype)
            for _ in range(count)]
  if np.issubdtype(dtype, np.integer):
    low = 0 if dtype is np.uint32 else -5
    return [rng.integers(low, 6, (2, 3)).astype(dtype) for _ in range(count)]
  return [rng.uniform(0.1, 0.9, (2, 3)).astype(dtype) for _ in range(count)]


def _bound(primitive, rng):
  """Operands of shape (2, 3) the primitive binds, a second draw of them, and its parameters; or None."""
  params = _PARAMS.get(primitive.name, {})
  for count, dtype, extra in itertools.product((1, 2, 3, 4), _DTYPES, ({}, {"accuracy": None})):
    draws = _drawn(count, dtype, rng), _drawn(count, dtype, rng)
    try:
      primitive.bind(*map(jnp.asarray, draws[0]), **params, **extra)
    except Exception:  # pylint: disable=broad-except
      continue
    return draws, {**params, **extra}
  return None


class PrimitiveAttributesTest(jtu.JaxTestCase):

  def test_an_elementwise_primitive_reads_each_output_entry_at_its_own_index(self):
    # An output entry is unchanged where every operand's entries at the other
    # indices are drawn again: it reads its own index's entries alone. A marked
    # primitive no operands bind fails, so the test reads every one it marks.
    rng = np.random.default_rng(0)
    unbound = []
    for primitive in _marked():
      made = _bound(primitive, rng)
      if made is None:
        unbound.append(primitive.name)
        continue
      (first, second), params = made

      def outputs(operands, primitive=primitive, params=params):
        outs = primitive.bind(*map(jnp.asarray, operands), **params)
        return [np.asarray(out) for out in (outs if primitive.multiple_results else [outs])]

      kept = outputs(first)
      for index in np.ndindex(2, 3):
        mixed = []
        for one, other in zip(first, second, strict=True):
          entry = np.array(other)
          entry[index] = one[index]
          mixed.append(entry)
        for whole, alone in zip(kept, outputs(mixed), strict=True):
          np.testing.assert_array_equal(whole[index], alone[index], err_msg=f"{primitive.name} at {index}")
    self.assertEqual(unbound, [])

  def test_a_reduction_contraction_or_permutation_is_not_elementwise(self):
    for primitive in (lax_internal.reduce_sum_p, lax_internal.dot_general_p, lax_internal.transpose_p):
      self.assertFalse(primitive.elementwise, primitive.name)

  def test_a_call_xla_inlines_is_marked_and_its_program_has_no_call(self):
    def body(x):
      return jnp.sin(x) * 2.0

    @jax.custom_jvp
    def custom_jvp(x):
      return body(x)
    custom_jvp.defjvp(lambda primals, tangents: (body(*primals), tangents[0]))

    @jax.custom_vjp
    def custom_vjp(x):
      return body(x)
    custom_vjp.defvjp(lambda x: (body(x), None), lambda _, g: (g,))

    @jax.custom_batching.custom_vmap
    def custom_vmap(x):
      return body(x)
    custom_vmap.def_vmap(lambda size, batched, x: (jax.vmap(body)(x), batched[0]))

    program = jax.make_jaxpr(body)(jnp.ones(4))
    calls = {
        "jit": jax.jit(body),
        "checkpoint": jax.checkpoint(body),
        "custom_jvp": custom_jvp,
        "custom_vjp": custom_vjp,
        "custom_vmap": custom_vmap,
        "closed_call": lambda x: core.closed_call_p.bind(x, call_jaxpr=program)[0],
    }
    for name, call in calls.items():
      with self.subTest(name):
        def outer(x):
          return call(x) + 1.0
        jaxpr = jax.make_jaxpr(outer)(jnp.ones(4))
        (called,) = [e for e in jaxpr.eqns if e.primitive is not lax_internal.add_p]
        self.assertTrue(called.primitive.inlined_call, called.primitive.name)
        compiled = jax.jit(outer).lower(jnp.ones(4)).compile().as_text()
        self.assertNotIn(" call(", compiled)

  def test_a_transparent_calls_transformation_is_its_programs(self):
    body = lambda x: jnp.sin(x) * x

    @jax.custom_jvp
    def custom_jvp(x):
      return body(x)
    custom_jvp.defjvp(lambda primals, tangents: (body(*primals), tangents[0]))

    program = jax.make_jaxpr(body)(jnp.ones(4))
    calls = {
        "jit": (jax.jit(body), True),
        "closed_call": (lambda x: core.closed_call_p.bind(x, call_jaxpr=program)[0], True),
        "checkpoint": (jax.checkpoint(body), False),
        "custom_jvp": (custom_jvp, False),
    }
    x, t = jnp.linspace(0.5, 2.0, 4), jnp.linspace(1.0, -1.0, 4)
    expected = jax.jvp(body, (x,), (t,))
    for name, (call, transparent) in calls.items():
      with self.subTest(name):
        (called,) = jax.make_jaxpr(call)(x).eqns
        self.assertEqual(called.primitive.transparent_call, transparent)
        self.assertTrue(called.primitive.inlined_call)
        if transparent:
          self.assertAllClose(jax.jvp(call, (x,), (t,)), expected)
    # A rule of its own is what keeps a call from being transparent: here a tangent passed through.
    self.assertAllClose(jax.jvp(custom_jvp, (x,), (t,))[1], t)

  def test_a_made_call_primitive_is_inlined_and_not_transparent(self):
    marked = jex.core.primitives.create_call_primitive("marked_call")
    program = jax.make_jaxpr(lambda x: jnp.cos(x) + 1.0)(jnp.ones(4))
    self.assertTrue(marked.inlined_call)
    self.assertFalse(marked.transparent_call)
    compiled = jax.jit(lambda x: marked.bind(x, call_jaxpr=program)[0]).lower(jnp.ones(4)).compile()
    self.assertNotIn(" call(", compiled.as_text())

  def test_the_primitives_an_analysis_dispatches_on_are_exported(self):
    from jax._src import callback, custom_batching, debugging, ffi, pjit, shard_map
    from jax._src.lax.control_flow import conditionals
    from jax._src.pallas import primitives as pallas_primitives
    from jax._src.random import core as random_core, prng
    from jax._src.state import primitives as state_primitives
    import jax.experimental.pallas as pl

    exported = jex.core.primitives
    for name, primitive in {
        "custom_vmap_p": custom_batching.custom_vmap_p, "shard_map_p": shard_map.shard_map_p,
        "random_wrap_p": prng.random_wrap_p, "random_unwrap_p": prng.random_unwrap_p,
        "random_clone_p": random_core.random_clone_p,
        "ragged_dot_general_p": lax_internal.ragged_dot_general_p,
        "pure_callback_p": callback.pure_callback_p, "io_callback_p": callback.io_callback_p,
        "debug_callback_p": debugging.debug_callback_p, "debug_print_p": debugging.debug_print_p,
        "ffi_call_p": ffi.ffi_call_p, "platform_index_p": conditionals.platform_index_p,
        "reshard_p": pjit.reshard_p, "addupdate_p": state_primitives.addupdate_p}.items():
      with self.subTest(name):
        self.assertIs(getattr(exported, name), primitive)
    self.assertIs(pl.program_id_p, pallas_primitives.program_id_p)
    self.assertIs(pl.num_programs_p, pallas_primitives.num_programs_p)
    self.assertIs(pl.load_p, pallas_primitives.load_p)
    self.assertIs(pl.swap_p, pallas_primitives.swap_p)

  def test_a_primitive_states_the_class_its_batching_rule_serves(self):
    x = jnp.arange(12.0).reshape(3, 4)
    key = jax.random.key(0)

    def primitives(program):
      found = set()
      def walk(jaxpr):
        for e in jaxpr.eqns:
          found.add(e.primitive)
          for inner in core.jaxprs_in_params(e.params):
            walk(inner)
      walk(jax.make_jaxpr(program)().jaxpr)
      return found

    programs = {
        "positionwise": [lambda: jnp.sin(x), lambda: x + x, lambda: jax.random.split(key),
                         lambda: jax.random.bits(key, (4,)), lambda: jax.lax.bitcast_convert_type(x, jnp.int32)],
        "reduces": [lambda: jnp.sum(x, axis=1), lambda: jnp.max(x), lambda: jnp.argmax(x, axis=0)],
        "cumulative": [lambda: jnp.cumsum(x, axis=1), lambda: jax.lax.cummax(x, axis=0),
                       lambda: jax.lax.cumlogsumexp(x, axis=1)],
        "scatters": [lambda: x.at[0].set(1.0), lambda: x.at[1].add(1.0), lambda: x.at[2].max(1.0)],
    }
    classes = tuple(programs)
    for fact, made in programs.items():
      for program in made:
        found = primitives(program)
        stated = [p for p in found if getattr(p, fact)]
        with self.subTest(fact=fact, primitives=sorted(p.name for p in found)):
          self.assertTrue(stated)
          # Each class is its rule's alone: no primitive states two.
          for primitive in stated:
            self.assertEqual([c for c in classes if getattr(primitive, c)], [fact])
    # A primitive whose batching rule is its own states none of them: a dot, a gather.
    for program, name in ((lambda: x @ x.T, "dot_general"), (lambda: x[jnp.array([0, 2])], "gather")):
      (primitive,) = [p for p in primitives(program) if p.name == name]
      with self.subTest(primitive=name):
        self.assertFalse(any(getattr(primitive, c) for c in classes))

  def test_a_rearranging_primitive_only_moves_entries(self):
    # Distinct odd values with no exact sum or product among them in float32: every output entry of a
    # rearrangement is one of them, or the pad's fill (-7.0), which no input holds.
    x = (jnp.arange(24.0) * 2 + 1).reshape(2, 3, 4) * np.float32(1.0000001)
    programs = [
        lambda: x.reshape(6, 4), lambda: jnp.squeeze(x[:1], 0), lambda: jnp.broadcast_to(x, (2, 2, 3, 4)),
        lambda: jnp.transpose(x, (2, 0, 1)), lambda: jnp.flip(x, 1), lambda: x[:, 1:3],
        lambda: jax.lax.dynamic_slice(x, (0, 1, 1), (1, 2, 2)),
        lambda: jax.lax.dynamic_update_slice(x, x[:1, :1, :1] * 0 + x[1, 2, 3], (0, 0, 0)),
        lambda: x[:, jnp.array([2, 0])], lambda: jnp.concatenate([x, x], 1),
        lambda: jax.lax.pad(x, np.float32(-7.0), [(1, 0, 0), (0, 1, 1), (0, 0, 0)]),
        lambda: jax.lax.split(x, (2, 2), 2)[1],
    ]
    allowed = set(np.asarray(x).ravel().tolist()) | {-7.0}

    def primitives(program):
      found = []
      def walk(jaxpr):
        for e in jaxpr.eqns:
          found.append(e.primitive)
          for inner in core.jaxprs_in_params(e.params):
            walk(inner)
      walk(jax.make_jaxpr(program)().jaxpr)
      return found

    seen = set()
    for program in programs:
      rearranging = [p for p in primitives(program) if p.rearranges]
      with self.subTest(primitives=[p.name for p in rearranging]):
        self.assertTrue(rearranging)
        self.assertLessEqual(set(np.asarray(jax.jit(program)()).ravel().tolist()), allowed)
        seen |= set(rearranging)
    self.assertEqual({p.name for p in seen},
                     {"reshape", "squeeze", "broadcast_in_dim", "transpose", "rev", "slice", "dynamic_slice",
                      "dynamic_update_slice", "gather", "concatenate", "pad", "split"})
    # Arithmetic and rounding are not rearrangements: an add, a convert, a dot.
    for program in (lambda: x + 1, lambda: x.astype(jnp.bfloat16), lambda: x.reshape(6, 4) @ x.reshape(4, 6)):
      for primitive in primitives(program):
        if primitive.name in ("add", "convert_element_type", "dot_general"):
          self.assertFalse(primitive.rearranges, primitive.name)

  def test_a_loop_or_branch_is_not_an_inlined_call(self):
    jaxpr = jax.make_jaxpr(lambda x: jax.lax.fori_loop(0, 3, lambda _, y: y * 2, x))(1.0)
    self.assertFalse(jaxpr.eqns[-1].primitive.inlined_call)
    jaxpr = jax.make_jaxpr(lambda p, x: jax.lax.cond(p, jnp.sin, jnp.cos, x))(True, 1.0)
    self.assertFalse(jaxpr.eqns[-1].primitive.inlined_call)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
