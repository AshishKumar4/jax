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
"""A float32 to float4_e2m1fn conversion rounds once, inside a loop XLA:CPU hoists as outside it.

XLA:CPU compiles a small while loop it hoists (SmallWhileLoopHoistingPass) through its legacy elemental emitter,
which converted through float16 rounded to nearest: -0.25004873 became float16 -0.25, a tie float4_e2m1fn breaks
to -0.0, where the value itself rounds to -0.5.
"""

from absl.testing import absltest
import ml_dtypes
import numpy as np

import jax
from jax import lax
import jax.numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class F4E2M1FNConvertTest(jtu.JaxTestCase):

  def test_a_scanned_conversion_rounds_once(self):
    values = np.random.default_rng(1).uniform(-6.5, 6.5, size=(2, 8, 64)).astype(np.float32)
    values[0, 0, :4] = [-0.25004873, 0.25004873, 0.7500305, -1.2500610]
    expected = values.astype(ml_dtypes.float4_e2m1fn).astype(np.float32)

    def body(carry, x):
      return carry, (x / carry).astype(jnp.float4_e2m1fn).astype(jnp.float32)

    scanned = jax.jit(lambda c, xs: lax.scan(body, c, xs)[1])(jnp.float32(1), values)
    np.testing.assert_array_equal(scanned, expected)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
