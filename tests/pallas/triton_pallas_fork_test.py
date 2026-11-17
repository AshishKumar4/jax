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

  @parameterized.parameters(
      (jnp.float32, 0), (jnp.float32, 1), (jnp.int32, 1), (jnp.bool_, 0)
  )
  def test_a_unit_slice_of_a_value_is_its_element(self, dtype, axis):
    # One element along one axis, exactly: a NaN and a negative zero among a
    # float's, a boolean's own value.
    x = jax.random.normal(jax.random.key(0), (16, 16), jnp.float32)
    x = x.at[3, 5].set(jnp.nan).at[5, 3].set(-0.0)
    x = (x > 0) if dtype == jnp.bool_ else x.astype(dtype)
    index = 5

    def kernel(x_ref, out_ref):
      value = x_ref[...]
      out_ref[...] = (
          lax.slice_in_dim(value, index, index + 1, axis=axis)
      )

    shape = (1, 16) if axis == 0 else (16, 1)
    out = self.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(shape, dtype), grid=(1,)
    )(x)
    expected = lax.slice_in_dim(x, index, index + 1, axis=axis)
    np.testing.assert_array_equal(
        np.asarray(out).view(np.uint8), np.asarray(expected).view(np.uint8)
    )

  def test_a_booleans_any_and_all_reduce(self):
    x = jax.random.normal(jax.random.key(0), (16, 16), jnp.float32) > 1.5

    def kernel(x_ref, any_ref, all_ref):
      value = x_ref[...]
      any_ref[...] = jnp.any(value, axis=1)
      all_ref[...] = jnp.all(value, axis=0)

    vector = jax.ShapeDtypeStruct((16,), jnp.bool_)
    found_any, found_all = self.pallas_call(
        kernel, out_shape=(vector, vector), grid=(1,)
    )(x)
    np.testing.assert_array_equal(found_any, np.any(np.asarray(x), axis=1))
    np.testing.assert_array_equal(found_all, np.all(np.asarray(x), axis=0))

  def test_a_hijax_primitive_lowers_as_its_expansion(self):
    from jax._src import hijax

    class Square(hijax.HiPrim):

      def __init__(self, in_aval):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return x * x

    x = jax.random.normal(jax.random.key(0), (16, 16), jnp.float32)

    def kernel(x_ref, out_ref):
      value = x_ref[...]
      out_ref[...] = Square(jax.typeof(value))(value)

    out = self.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), grid=(1,)
    )(x)
    np.testing.assert_array_equal(out, x * x)


if __name__ == "__main__":
  absltest.main()
