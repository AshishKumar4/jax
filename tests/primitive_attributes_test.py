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

"""Primitive.elementwise and Primitive.inlined_call hold of what they mark."""

import itertools

from absl.testing import absltest
import numpy as np

import jax
from jax._src import ad_util
from jax._src import core
from jax._src import test_util as jtu
from jax._src.lax import lax as lax_internal
from jax._src.lax import special
from jax._src.random import threefry2x32
import jax.numpy as jnp

jax.config.parse_flags_with_absl()

# The parameters a marked primitive binds with that its operands do not imply.
_PARAMS = {
    "integer_pow": {"y": 3},
    "convert_element_type": {"new_dtype": np.dtype(np.float32), "weak_type": False, "sharding": None},
    "reduce_precision": {"exponent_bits": 5, "mantissa_bits": 10},
}

_DTYPES = (np.float32, np.int32, np.uint32, np.bool_, np.complex64)


def _marked():
  modules = (lax_internal, special, ad_util, threefry2x32)
  found = {value for module in modules for value in vars(module).values()
           if isinstance(value, core.Primitive) and value.elementwise}
  return sorted(found, key=lambda p: p.name)


def _operands(primitive, rng):
  """A shape (2, 3) operand set the primitive binds, its outputs, or None."""
  params = _PARAMS.get(primitive.name, {})
  extras = ({}, {"accuracy": None})
  for count, dtype, extra in itertools.product((1, 2, 3, 4), _DTYPES, extras):
    if dtype is np.bool_:
      operands = [rng.random((2, 3)) > 0.5 for _ in range(count)]
    elif dtype is np.complex64:
      operands = [(rng.uniform(0.1, 0.9, (2, 3)) + 1j * rng.uniform(0.1, 0.9, (2, 3))).astype(dtype)
                  for _ in range(count)]
    elif np.issubdtype(dtype, np.integer):
      operands = [rng.integers(0 if dtype is np.uint32 else -5, 6, (2, 3)).astype(dtype) for _ in range(count)]
    else:
      operands = [rng.uniform(0.1, 0.9, (2, 3)).astype(dtype) for _ in range(count)]
    if primitive.name == "select_n":
      operands[0] = operands[0].astype(np.bool_) if count == 3 else operands[0]
    try:
      outs = primitive.bind(*map(jnp.asarray, operands), **params, **extra)
    except Exception:  # pylint: disable=broad-except
      continue
    return operands, {**params, **extra}, outs
  return None


class PrimitiveAttributesTest(jtu.JaxTestCase):

  def test_an_elementwise_primitive_reads_each_output_entry_at_its_own_index(self):
    # Each output entry equals the primitive applied to its operands' entries at
    # that index alone (bound on scalars); a marked primitive no operands bind
    # fails, so the test reads every one it marks.
    rng = np.random.default_rng(0)
    unbound = []
    for primitive in _marked():
      made = _operands(primitive, rng)
      if made is None:
        unbound.append(primitive.name)
        continue
      operands, params, outs = made
      outs = outs if primitive.multiple_results else [outs]
      for index in np.ndindex(2, 3):
        alone = primitive.bind(*(jnp.asarray(o[index]) for o in operands), **params)
        alone = alone if primitive.multiple_results else [alone]
        for whole, entry in zip(outs, alone, strict=True):
          np.testing.assert_array_equal(np.asarray(whole)[index], np.asarray(entry),
                                        err_msg=f"{primitive.name} at {index}")
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

    program = jax.make_jaxpr(body)(jnp.ones(4))
    calls = {
        "jit": jax.jit(body),
        "checkpoint": jax.checkpoint(body),
        "custom_jvp": custom_jvp,
        "custom_vjp": custom_vjp,
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

  def test_a_loop_or_branch_is_not_an_inlined_call(self):
    jaxpr = jax.make_jaxpr(lambda x: jax.lax.fori_loop(0, 3, lambda _, y: y * 2, x))(1.0)
    self.assertFalse(jaxpr.eqns[-1].primitive.inlined_call)
    jaxpr = jax.make_jaxpr(lambda p, x: jax.lax.cond(p, jnp.sin, jnp.cos, x))(True, 1.0)
    self.assertFalse(jaxpr.eqns[-1].primitive.inlined_call)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
