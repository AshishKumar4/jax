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

"""An operation labeled `xla_no_contract` rounds on its own on the CPU, its
unlabeled twins contracted as before, and the label leaves fusion as it was."""

import re

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax.experimental.xla_metadata import set_xla_metadata
import jax.extend.backend
import jax.numpy as jnp
import numpy as np


def _operands():
  x = jax.random.normal(jax.random.key(0), (4096,), jnp.float32)
  y = jax.random.normal(jax.random.key(1), (4096,), jnp.float32)
  z = jax.random.normal(jax.random.key(2), (4096,), jnp.float32)
  return x, y, z


def _contracted(x, y, z):
  return x * y + z


def _labeled(x, y, z):
  with set_xla_metadata(xla_no_contract=True):
    product = x * y
  return product + z


def _both(x, y, z):
  # The labeled product and an unlabeled one of the same operands, each read by
  # an add: one program, so common subexpressions could join them.
  with set_xla_metadata(xla_no_contract=True):
    kept = x * y
  return kept + z, x * y + z


@jtu.with_config(jax_enable_compilation_cache=False)
class NoContractTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("the CPU's contraction")
    if jax.extend.backend.no_contract_attribute() is None:
      self.skipTest("this jaxlib's compilers do not honour the label")
    x, y, z = _operands()
    rounded = np.asarray(x) * np.asarray(y) + np.asarray(z)
    if np.array_equal(np.asarray(jax.jit(_contracted)(x, y, z)), rounded):
      self.skipTest("this CPU contracts nothing: no fused multiply-add")

  def test_a_labeled_product_rounds_before_its_sum(self):
    x, y, z = _operands()
    rounded = np.asarray(x) * np.asarray(y) + np.asarray(z)
    np.testing.assert_array_equal(jax.jit(_labeled)(x, y, z), rounded)

  def test_a_labeled_product_and_its_unlabeled_twin_each_keep_their_rounding(
      self):
    x, y, z = _operands()
    rounded = np.asarray(x) * np.asarray(y) + np.asarray(z)
    kept, fused = jax.jit(_both)(x, y, z)
    np.testing.assert_array_equal(kept, rounded)
    np.testing.assert_array_equal(fused, jax.jit(_contracted)(x, y, z))

  def test_the_label_leaves_fusion_as_it_was(self):
    x, y, z = _operands()

    def optimized(fn):
      # The compiled module's computations, their labels, source metadata and
      # module name dropped, and the tables of the source's stack frames after
      # its entry computation, the last, which a label's context adds a frame
      # to: what is fused, and how, alone.
      text = jax.jit(fn).lower(x, y, z).compile().as_text()
      text = text[:text.index("\n}", text.index("\nENTRY ")) + 2]
      text = re.sub(r",? ?(?:frontend_attributes|metadata)=\{[^}]*\}", "", text)
      return re.sub(r"^HloModule \S+", "HloModule", text, flags=re.M)

    self.assertEqual(optimized(_labeled), optimized(_contracted))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
