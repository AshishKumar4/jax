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

import contextlib
import functools
import io
import sys

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import custom_call_targets
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.lib.mlir import ir
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

if sys.platform != "win32":
  from jax.experimental.pallas import triton as plgpu
  from jax._src.lib.triton import dialect as tt_dialect
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

  def test_a_tile_loaded_once_and_used_twice_is_the_same_pipelined(self):
    # An attention backward's dK: each step loads a query tile once, scales it
    # in registers for the scores and feeds it raw to the warpgroup MMA summing
    # into dK. Pipelined at two stages, the next step's copy must not overwrite
    # the tile that MMA still reads asynchronously, so the kernel computes at
    # two stages exactly what it computes at one, every run.
    self._require_compute_capability("9.0")
    block, steps, width = 64, 16, 64

    def kernel(q_ref, k_ref, dk_ref):
      k = k_ref[...]

      def step(i, dk):
        q = q_ref[pl.ds(i * block, block), :]
        scores = plgpu.dot((q * 0.5).astype(q.dtype), k, trans_b=True)
        return dk + plgpu.dot(scores.astype(q.dtype), q, trans_a=True)

      dk_ref[...] = lax.fori_loop(
          0, steps, step, jnp.zeros((block, width), jnp.float32))

    q = jax.random.normal(jax.random.key(0), (steps * block, width), jnp.bfloat16)
    k = jax.random.normal(jax.random.key(1), (block, width), jnp.bfloat16)

    def call(stages):
      return self.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((block, width), jnp.float32),
          grid=(1,),
          compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=stages),
      )(q, k)

    unpipelined = call(1)
    for _ in range(8):
      self.assertArraysEqual(call(2), unpipelined)


intx = dtypes.default_int_dtype()


@functools.partial(
    jax.jit, static_argnames=["bm", "bn", "gm", "bk", "interpret", "debug"]
)
def matmul(x, y, *, bm, bn, gm, bk, interpret, debug=False):
  m, n, k = x.shape[0], y.shape[1], x.shape[1]

  @functools.partial(
      pl.pallas_call,
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      grid=pl.cdiv(m, bm) * pl.cdiv(n, bn),
      interpret=interpret,
      debug=debug,
      compiler_params=plgpu.CompilerParams(),
  )
  def matmul_kernel(x_ref, y_ref, o_ref):
    pid = pl.program_id(axis=0).astype(intx)
    num_pid_m = m // bm
    num_pid_n = n // bn
    num_pid_in_group = gm * num_pid_n
    group_id = lax.div(pid, num_pid_in_group)
    first_pid_m = group_id * gm
    group_size_m = jnp.minimum(num_pid_m - first_pid_m, gm)
    pid_m = first_pid_m + lax.rem(pid, group_size_m)
    pid_n = lax.div(lax.rem(pid, num_pid_in_group), group_size_m)
    idx_m = pid_m * bm + jnp.arange(bm)
    idx_n = pid_n * bn + jnp.arange(bn)
    idx_m = plgpu.max_contiguous(pl.multiple_of(idx_m, bm), bm)
    idx_n = plgpu.max_contiguous(pl.multiple_of(idx_n, bn), bn)
    acc = jnp.zeros((bm, bn), dtype=jnp.float32)

    def body(i, acc):
      idx_k = i * bk + jnp.arange(bk)
      x_idx = (
          lax.broadcast_in_dim(idx_m, (bm, bk), (0,)),
          lax.broadcast_in_dim(idx_k, (bm, bk), (1,)),
      )
      y_idx = (
          lax.broadcast_in_dim(idx_k, (bk, bn), (0,)),
          lax.broadcast_in_dim(idx_n, (bk, bn), (1,)),
      )
      x_block, y_block = x_ref[x_idx], y_ref[y_idx]
      out = plgpu.dot(x_block, y_block)
      return acc + out

    acc = lax.fori_loop(0, k // bk, body, acc).astype(o_ref.dtype)
    o_idx = (
        lax.broadcast_in_dim(idx_m, (bm, bn), (0,)),
        lax.broadcast_in_dim(idx_n, (bm, bn), (1,)),
    )
    o_ref[o_idx] = acc

  return matmul_kernel(x, y)


class TritonKernelLocationsTest(PallasBaseTest):

  @parameterized.parameters(False, True)
  def test_a_kernel_serializes_locations_only_when_debugged(self, debug):
    # A reduction's combiner is traced once a process, so its locations named
    # the first kernel to lower it in every later one: a kernel serialized
    # apart across processes and call orders, its program with it.
    if jtu.test_device_matches(["cpu"]):
      self.skipTest("Lowers a Triton kernel only for a GPU")

    def body(x_ref, o_ref):
      o_ref[...] = jnp.sum(x_ref[...], axis=0, keepdims=True)

    f = self.pallas_call(
        body, out_shape=jax.ShapeDtypeStruct((1, 128), jnp.float32),
        debug=debug)
    with contextlib.redirect_stdout(io.StringIO()):
      lowered = jax.jit(f).lower(jnp.ones((16, 128), jnp.float32))
    kernels = []

    def find(op):
      if (op.name == "stablehlo.custom_call"
          and op.attributes["call_target_name"].value
          == custom_call_targets.TRITON):
        config = ir.DictAttr(op.attributes["mhlo.backend_config"])
        kernels.append(ir.StringAttr(config["ir"]).value_bytes)
      return ir.WalkResult.ADVANCE

    lowered.compiler_ir().operation.walk(find)
    [kernel] = kernels
    ctx = ir.Context()
    tt_dialect.register_dialect(ctx)
    known = []

    def located(op):
      if str(op.location) != "loc(unknown)":
        known.append(str(op.location))
      return ir.WalkResult.ADVANCE

    with ctx:
      ir.Module.parse(kernel).operation.walk(located)
    self.assertEqual(bool(known), debug, known[:3])


class TritonDialectLookupTest(jtu.JaxTestCase):

  def test_a_triton_operation_finds_its_class_without_searching(self):
    # MLIR finds a `tt` operation's Python class under a dialect search prefix
    # and keeps the module: lowering a Triton kernel imports nothing more, at
    # its first operation or its thousandth.
    try:
      from jax._src.lib import triton  # pytype: disable=import-error
    except ImportError:
      self.skipTest("this jaxlib has no Triton dialect")
    from jax._src.interpreters import mlir
    from jax._src.lib.mlir import ir
    from jax._src.pallas.triton import lowering

    searched = []

    class Recorder:

      def find_spec(self, fullname, path=None, target=None):
        searched.append(fullname)
        return None

    def build():
      with lowering._new_ir_context(), ir.Location.unknown():
        module = ir.Module.parse(
            "module {\n"
            "  tt.func @kernel(%arg0: tensor<4xf32>) -> tensor<4xf32> {\n"
            "    tt.return %arg0 : tensor<4xf32>\n"
            "  }\n"
            "}\n")
        # Each operation's view is its class's, which MLIR looks up.
        return [type(op.operation.opview) for op in module.body.operations]

    del mlir, triton
    first = build()
    sys.meta_path.insert(0, Recorder())
    try:
      again = [build() for _ in range(3)]
    finally:
      sys.meta_path.pop(0)
    self.assertEqual(searched, [])
    self.assertEqual(first[0].__name__, "FuncOp")
    self.assertTrue(all(views == first for views in again))


if __name__ == "__main__":
  absltest.main()
