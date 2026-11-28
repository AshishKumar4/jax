# Copyright 2021 The JAX Authors.
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

"""The raw facts a compiled executable states for its readers: XLA's buffer
assignment and which call argument each entry parameter holds."""

import re

from absl.testing import absltest
import jax
from jax import lax
from jax._src import test_util as jtu
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp


class CompiledFactsTest(jtu.JaxTestCase):

  def test_each_entry_parameter_names_the_argument_it_holds(self):
    # x is unused and pruned, so the module's one parameter holds argument 1.
    pruned = jax.jit(lambda x, y: y * 2).lower(1.0, 2.0).compile()
    self.assertEqual(pruned.entry_arguments(), (1,))
    kept = jax.jit(lambda x, y: y * 2, keep_unused=True).lower(1.0, 2.0)
    self.assertEqual(kept.compile().entry_arguments(), (0, 1))

  def test_a_closed_over_constant_holds_no_argument(self):
    constant = jnp.arange(4.0)
    compiled = jax.jit(lambda x: x + constant).lower(jnp.ones(4)).compile()
    # The constant, where it is a parameter, holds no argument of the call.
    arguments = compiled.entry_arguments()
    self.assertEqual([a for a in arguments if a is not None], [0])

  @jtu.run_on_devices("cpu")
  def test_the_buffer_assignment_is_xlas_serialized(self):
    compiled = jax.jit(lambda x: x @ x.T).lower(jnp.ones((8, 4))).compile()
    assignment = compiled.buffer_assignment()
    self.assertIsInstance(assignment, bytes)
    self.assertNotEmpty(assignment)


class CheckpointNameTest(jtu.JaxTestCase):
  """A checkpoint name is its value's defining op's frontend attribute, which
  XLA keeps on the instruction defining the value's buffer."""

  def named_ops(self, f, *args):
    text = jax.jit(f).lower(*args).as_text()
    return [re.search(r'= "?(\w+\.\w+)', line).group(1)
            for line in text.splitlines() if 'checkpoint_name = "' in line]

  def test_a_name_tags_its_values_producer_through_a_reshape(self):
    named = self.named_ops(
        lambda x: checkpoint_name((x @ x).reshape(-1), "hidden"),
        jnp.ones((4, 4)))
    self.assertEqual(named, ["stablehlo.dot_general", "stablehlo.reshape"])

  def test_a_name_tags_its_values_producer_through_a_residual_anchor(self):
    # JAX anchors a saved residual with a reduce-precision to its own type,
    # which XLA may remove once fused: the value's producer carries the name.
    def loss(x):
      saved = jax.checkpoint(
          lambda x: jnp.sin(checkpoint_name(jnp.tanh(x), "hidden")),
          policy=jax.checkpoint_policies.save_only_these_names("hidden"))
      return saved(x).sum()
    self.assertEqual(self.named_ops(jax.grad(loss), jnp.ones(4)),
                     ["stablehlo.tanh", "stablehlo.reduce_precision"])

  def test_a_name_stops_at_a_reduce_precision_that_rounds(self):
    rounded = lambda x: lax.reduce_precision(x, exponent_bits=5,
                                             mantissa_bits=10)
    named = self.named_ops(
        lambda x: checkpoint_name(rounded(jnp.tanh(x)), "hidden"),
        jnp.ones(4, jnp.float32))
    self.assertEqual(named, ["stablehlo.reduce_precision"])

  def test_a_name_on_one_result_of_a_variadic_op_tags_nothing(self):
    # sort's two results share one instruction: naming its op would name the
    # unnamed sibling's buffer too.
    def f(keys, values):
      _, sorted_values = lax.sort((keys, values), num_keys=1)
      return checkpoint_name(sorted_values, "values")
    self.assertEqual(self.named_ops(f, jnp.arange(8.0), jnp.arange(8.0)), [])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
