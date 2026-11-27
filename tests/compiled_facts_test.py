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

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
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


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
