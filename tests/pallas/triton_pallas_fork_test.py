# Copyright 2025 The JAX Authors.
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

import sys

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

if sys.platform != "win32":
  from jax.experimental.pallas import triton as plgpu
else:
  plgpu = None

config.parse_flags_with_absl()


@jtu.with_config(jax_traceback_filtering="off")
class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if jtu.test_device_matches(["cpu"]):
      if not self.INTERPRET:
        self.skipTest("On CPU the test works only in interpret mode")
    elif jtu.test_device_matches(["gpu"]):
      if (jtu.test_device_matches(["cuda"]) and
          not jtu.is_cuda_compute_capability_at_least("8.0")):
        self.skipTest("Only works on GPU with capability >= sm80")
      if plgpu is None:
        self.skipTest("plgpu not available on this platform")
      self.enter_context(
          jtu.ignore_warning(
              category=DeprecationWarning,
              message="The Pallas Triton backend is deprecated",
          )
      )
    else:
      self.skipTest("Test only works on CPU and GPU")

    super().setUp()

  def _require_compute_capability(self, min_version_str):
    if (jtu.test_device_matches(["cuda"])
        and not jtu.is_cuda_compute_capability_at_least(min_version_str)):
      self.skipTest(f"Requires CUDA compute capability >= {min_version_str}")

  def pallas_call(self, *args, **kwargs):
    # Force the use of the Triton backend.
    kwargs.setdefault("compiler_params", plgpu.CompilerParams())
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)


class TritonForkTest(PallasBaseTest):
  """Pallas Triton lowerings the fork carries (upstream removed its Triton tests, a67386b646)."""
  INTERPRET = False

  def test_a_product_beside_its_barriered_self_rounds_on_its_own(self):
    # x * y - barrier(x * y): each product rounds on its own, so the difference
    # is zero; contracted into an FMA it would be the product's rounding error.
    def kernel(x_ref, y_ref, out_ref):
      x, y = x_ref[...], y_ref[...]
      (rounded,) = lax.optimization_barrier((x * y,))
      out_ref[...] = x * y - rounded

    x = jax.random.normal(jax.random.key(0), (64, 64), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (64, 64), dtype=jnp.float32)
    out = self.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((64, 64), jnp.float32), grid=(1,)
    )(x, y)
    self.assertArraysEqual(out, jnp.zeros_like(out))


if __name__ == "__main__":
  absltest.main()
