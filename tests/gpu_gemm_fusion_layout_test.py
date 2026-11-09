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
"""XLA:GPU's GEMM fusion keeps a layout's permutation across the bitcasts around a Triton GEMM.

A transpose whose result the reduction reads in another layout, reshaped into a dot's operand: GEMM fusion
commuted the bitcast across the copy layout assignment put there and the Triton GEMM read its operand
permuted (upstream XLA PR #51630, jax-ml/jax#41952; Dew's input-side gradients, ledger 5275).
"""

from absl.testing import absltest
import numpy as np

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class GemmFusionLayoutTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("GEMM fusion is XLA:GPU's")

  def test_transpose_reduction_reshape_before_dot(self):
    rng = np.random.default_rng(0)
    p0 = rng.standard_normal((16, 16, 3)).astype(np.float32)
    p1 = rng.standard_normal((16, 32)).astype(np.float32)

    def f(p0, p1):
      reduced = jnp.transpose(p0, (2, 0, 1)).reshape(3, 4, 4, 4, 4).sum(axis=(2, 4))
      return jnp.dot(reduced.reshape(3, 16), p1,
                     precision=jax.lax.Precision.HIGHEST)

    expected = (np.transpose(p0.astype(np.float64), (2, 0, 1))
                .reshape(3, 4, 4, 4, 4).sum(axis=(2, 4)).reshape(3, 16)
                @ p1.astype(np.float64))
    np.testing.assert_allclose(jax.jit(f)(p0, p1), expected, rtol=1e-3, atol=1e-3)

  def test_a_column_major_dot_bitcast_into_its_epilogue_adds(self):
    # The input gradient of three projections sharing one input, as XLA fused it in Dew's decoder: the
    # dot written column-major ([64,24]{0,1}), bitcast to the row-major [24,64]{1,0} its two epilogue
    # addends hold. The Triton GEMM must add each element to its own addend, not its transpose's (Dew
    # ledger 5275: 0.52-1.03 relative error on an A100, right with Triton GEMM fusion off).
    rng = np.random.default_rng(0)
    first, second = (rng.standard_normal((24, 64)).astype(np.float32) for _ in range(2))
    weight = rng.standard_normal((64, 32)).astype(np.float32)
    gradient = rng.standard_normal((24, 32)).astype(np.float32)

    def f(first, second, weight, gradient):
      product = jnp.dot(gradient, weight.T, precision=jax.lax.Precision.HIGHEST)
      return product + first + second

    wide = [x.astype(np.float64) for x in (first, second, weight, gradient)]
    expected = wide[3] @ wide[2].T + wide[0] + wide[1]
    np.testing.assert_allclose(jax.jit(f)(first, second, weight, gradient), expected,
                               rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
