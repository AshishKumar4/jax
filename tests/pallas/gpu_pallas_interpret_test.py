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

import dataclasses
import functools
import threading
import math
from typing import Any
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.pallas import mpmd
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as mosaic_interpret
from jax._src.pallas.mosaic_gpu.interpret.params import force_gpu_interpret_mode
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams as InterpretParams
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


def _maybe_reverse(arg: tuple[Any], reverse: bool) -> tuple[Any]:
  if reverse:
    return tuple(reversed(arg))
  else:
    return arg


# TODO(nrink): Figure out how to safely run different instance of GPU
# interpret mode in parallel, and then remove this decorator.
@jtu.thread_unsafe_test_class()
class InterpretTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        jtu.ignore_warning(
            category=DeprecationWarning,
            message='jax.experimental.pallas.core_map is deprecated',
        )
    )
    try:
      # If an exception was thrown by a jitted computation during the
      # previous test, we observe/consume exception here to avoid propagating
      # it to the next test.
      jax.effects_barrier()
    except:
      pass
    mosaic_interpret.gpu_callbacks.reset_gpu_interpret_mode_state()

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    self.num_devices = jax.device_count()
    if self.num_devices > 1:
      self.skipTest(f'requires 1 device, found {self.num_devices}')

  def test_hopper_matmul(self):
    (m, n, k) = 512, 512, 512
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    a = jax.random.uniform(k1, (m, k), jnp.float16)
    b = jax.random.uniform(k2, (k, n), jnp.float16)
    c = jax.random.uniform(k3, (m, n), jnp.float32)

    spec = hopper_matmul_mgpu.TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=128,
        max_concurrent_steps=2,
        epi_tile_m=64,
        epi_tile_n=64,
        wg_dimension=hopper_matmul_mgpu.MatmulDimension.M,
    )

    device = jax.sharding.AbstractDevice(
        device_kind='NVIDIA H100 80GB HBM3',
        platform='gpu',
        num_cores=8,
    )
    with (jax.sharding.use_abstract_mesh(
              jax.sharding.AbstractMesh((), (), abstract_device=device)),
          force_gpu_interpret_mode(InterpretParams())):
      res = hopper_matmul_mgpu.matmul(a, b, c, spec).block_until_ready()
    expected = (
        jnp.dot(a, b, preferred_element_type=jnp.float32) + c)
    np.testing.assert_allclose(res, expected, rtol=5e-3)

  def test_interpret_pallas_call(self):
    def _kernel(o_ref):
      o_ref[0] = 42

    @jax.jit
    def kernel():
      return pl.pallas_call(
          _kernel,
          out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
          interpret=InterpretParams(detect_races=True),
      )()

    np.testing.assert_equal(kernel(), np.array([42], dtype=jnp.int32))

  @jtu.parameterized.parameters(range(1, 17))
  def test_interpret_core_map(self, num_threads: int):
    @pl.run_state
    def kernel(o_ref):
      mesh = plgpu.Mesh(num_threads=num_threads, thread_name='x')

      @pl.core_map(
          mesh,
          interpret=InterpretParams(detect_races=True),
      )
      def _():
        thread_idx = jax.lax.axis_index('x')
        o_ref[thread_idx] = thread_idx

    y = kernel(jnp.zeros((num_threads,), jnp.int32))
    np.testing.assert_equal(y, np.arange(num_threads, dtype=jnp.int32))

  def test_interpret_core_map_with_race(self):
    @pl.run_state
    def kernel(o_ref):
      mesh = plgpu.Mesh(num_threads=2, thread_name='x')

      @pl.core_map(
          mesh,
          interpret=InterpretParams(detect_races=True),
      )
      def _():
        thread_idx = jax.lax.axis_index('x')
        o_ref[...] = thread_idx

    with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
      kernel(jnp.zeros((), jnp.int32)).block_until_ready()

  @jtu.parameterized.parameters(range(1, 17))
  def test_interpret_kernel(self, num_threads):

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((num_threads,), jnp.int32),
        num_threads=num_threads,
        thread_name='x',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(o_ref):
      thread_idx = jax.lax.axis_index('x')
      o_ref[thread_idx] = thread_idx

    np.testing.assert_equal(jax.jit(_kernel)(), np.arange(num_threads))

  @jtu.parameterized.product(steps=(1, 2, 5), stages=(2, 3), carried=(False, True))
  def test_pipeline_prefetch_reads_each_steps_first_values_at_the_step_before(
      self, steps, stages, carried
  ):
    # Each step's first row read by `prefetch` at the step before it (the
    # first step's before the loop), the step's body reading it beside the
    # whole block; the body calls `boundary` before its last read.
    rows, width = 8, 128

    def body(o_ref, scratch):
      def step(indices, x_smem, *rest):
        if carried:
          total, prefetched, boundary = rest
        else:
          prefetched, boundary = rest
        (i,) = indices
        o_ref[i, 0] = prefetched
        following = boundary()
        o_ref[i, 1] = x_smem[0]
        if carried:
          return total + x_smem[rows - 1], following
        return following

      def prefetch(indices, x_smem):
        del indices
        return x_smem[0]

      out = plgpu.emit_pipeline(
          step,
          grid=(steps,),
          in_specs=[plgpu.BlockSpec((rows, width), lambda i: (i, 0))],
          max_concurrent_steps=stages,
          prefetch=prefetch,
          init_carry=jnp.zeros((width,), jnp.float32) if carried else None,
      )(scratch)
      if carried:
        o_ref[steps, 0] = out

    def kernel(x_ref, o_ref):
      body(o_ref, x_ref)

    x = jnp.arange(steps * rows * width, dtype=jnp.float32).reshape(
        steps * rows, width
    )
    out = plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((steps + 1, 2, width), jnp.float32),
        interpret=InterpretParams(),
    )(x)
    firsts = x[::rows]
    np.testing.assert_array_equal(out[:steps, 0], firsts)
    np.testing.assert_array_equal(out[:steps, 1], firsts)
    if carried:
      np.testing.assert_array_equal(out[steps, 0], x[rows - 1::rows].sum(0))

  def test_pipeline_prefetch_body_without_boundary_is_refused(self):
    def kernel(x_ref, o_ref):
      del o_ref
      plgpu.emit_pipeline(
          lambda indices, x_smem, prefetched, boundary: prefetched,
          grid=(2,),
          in_specs=[plgpu.BlockSpec((8, 128), lambda i: (i, 0))],
          max_concurrent_steps=2,
          prefetch=lambda indices, x_smem: x_smem[0],
      )(x_ref)

    with self.assertRaisesRegex(ValueError, "calls its boundary"):
      plgpu.kernel(
          kernel,
          out_type=jax.ShapeDtypeStruct((8,), jnp.float32),
          interpret=InterpretParams(),
      )(jnp.zeros((16, 128), jnp.float32))

  def _ragged_block_pipeline(self, x, start, rows):
    # A block of `rows` rows from row `start` of `x`, which may reach past
    # its last row, through a pipeline that reads past its operand as zeros
    # (`plgpu.BlockSpec`'s default `OOBFillMode.ZEROS`), copied out whole.
    def kernel(x_ref, o_ref):
      def step(indices, x_smem):
        del indices
        o_ref[...] = x_smem[...]

      plgpu.emit_pipeline(
          step,
          grid=(1,),
          in_specs=[plgpu.BlockSpec(
              # The start traced, as a tile's rows are: a static one past
              # the operand is refused at indexing.
              (pl.Element(rows), x.shape[1]), lambda i: (i + start, 0)
          )],
          max_concurrent_steps=1,
      )(x_ref)

    return functools.partial(
        plgpu.kernel,
        kernel,
        out_type=jax.ShapeDtypeStruct((rows, x.shape[1]), x.dtype),
    )

  def test_pipeline_reads_a_ragged_last_block_past_its_operand_as_zeros(self):
    # Rows 200 to 263 of a 256-row operand: the 56 inside it, then zeros.
    x = jnp.arange(256 * 128, dtype=jnp.float32).reshape(256, 128) + 1
    out = self._ragged_block_pipeline(x, 200, 64)(
        interpret=InterpretParams()
    )(x)
    np.testing.assert_array_equal(out[:56], x[200:])
    np.testing.assert_array_equal(out[56:], np.zeros((8, 128), np.float32))

  def test_a_ragged_pipeline_block_lowers_for_an_a100s_cp_async(self):
    # The same block lowered for an A100 (cp.async, a compile-only device):
    # each transfer's source size its bytes inside the operand.
    x = jax.ShapeDtypeStruct((256, 128), jnp.float32)
    device = jax.sharding.AbstractDevice(
        device_kind='NVIDIA A100-SXM4-80GB',
        platform='cuda',
        num_cores=None,
        compute_capability='8.0',
    )
    with jax.sharding.use_abstract_mesh(
        jax.sharding.AbstractMesh((), (), abstract_device=device)
    ):
      call = self._ragged_block_pipeline(x, 200, 64)()
      jax.jit(call).trace(x).lower(lowering_platforms=('cuda',))

  def test_layout_cast(self):
    # the layout_cast is a no-op in interpret mode
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 64), jnp.float32),
        interpret=InterpretParams(),
    )
    def _kernel(x_ref, o_ref):
      o_ref[...] = plgpu.layout_cast(x_ref[...], plgpu.Layout.WGMMA)

    x = jnp.arange(128 * 64, dtype=jnp.float32).reshape(128, 64)
    np.testing.assert_array_equal(_kernel(x), x)

  @jtu.parameterized.parameters(True, False)
  def test_fresh_axis_index_is_axis_index(self, functional):
    def body(o_ref):
      a, b = jax.lax.axis_index('a'), jax.lax.axis_index('b')
      o_ref[a, b] = plgpu.fresh_axis_index('a') * 10 + plgpu.fresh_axis_index('b')

    out = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((2, 3), jnp.int32),
        grid=(2, 3),
        grid_names=('a', 'b'),
        interpret=InterpretParams(functional=functional),
    )()
    np.testing.assert_array_equal(
        out, np.arange(2)[:, None] * 10 + np.arange(3)[None, :]
    )

  @jtu.parameterized.parameters(True, False)
  def test_functional_interpreter_agrees_with_simulated_memory(self, grid):
    # One thread a block: a grid of blocks, each copying its slice through
    # SMEM, scaled, with a loop and a branch.
    def body(x_ref, o_ref, smem):
      i = jax.lax.axis_index('b') if grid else 0
      smem[...] = plgpu.load(x_ref.at[pl.ds(i * 16, 16)])

      def step(j, total):
        return total + jnp.where(j % 2 == 0, smem[...], -smem[...])

      o_ref[pl.ds(i * 16, 16)] = jax.lax.fori_loop(0, 3, step, smem[...] * 0)

    def kernel(functional):
      return functools.partial(
          plgpu.kernel,
          out_type=jax.ShapeDtypeStruct((64, 8), jnp.float32),
          grid=(4,) if grid else (),
          grid_names=('b',) if grid else (),
          scratch_types=dict(smem=plgpu.SMEM((16, 8), jnp.float32)),
          interpret=InterpretParams(functional=functional),
      )(body)

    x = jnp.arange(64 * 8, dtype=jnp.float32).reshape(64, 8)
    functional = jax.jit(kernel(True))(x)
    simulated = jax.jit(kernel(False))(x)
    np.testing.assert_array_equal(functional, simulated)

  def test_functional_interpreter_leaves_unwritten_memory_nan(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((8, 8), jnp.float32),
        interpret=InterpretParams(functional=True),
    )
    def _kernel(o_ref):
      o_ref[0] = jnp.zeros((8,), jnp.float32)

    out = jax.jit(_kernel)()
    np.testing.assert_array_equal(out[0], 0)
    self.assertTrue(np.isnan(out[1:]).all())

  def test_functional_interpreter_raises_on_an_out_of_bounds_read(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((8,), jnp.float32),
        interpret=InterpretParams(functional=True),
    )
    def _kernel(x_ref, n_ref, o_ref):
      o_ref[...] = x_ref[pl.ds(n_ref[0], 8)]

    with self.assertRaisesRegex(Exception, 'Out-of-bounds'):
      jax.block_until_ready(jax.jit(_kernel)(
          jnp.zeros((8,), jnp.float32), jnp.array([4], jnp.int32)))

  def test_functional_interpreter_copies_through_tma_past_gmem_as_the_hardware(self):
    # A TMA load past its GMEM source reads zeros; a TMA store past its GMEM
    # destination drops the overhang.
    x = jnp.arange(6, dtype=jnp.int32).reshape((3, 2))

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((3, 2), jnp.int32),
        interpret=InterpretParams(functional=True),
        scratch_types=dict(barrier=plgpu.Barrier(), smem=plgpu.SMEM((4, 2), jnp.int32)),
    )
    def _kernel(start_ref, in_gmem, out_gmem, barrier, smem):
      # Starts read at run time: a static start past the end is refused when
      # the kernel is traced.
      plgpu.copy_gmem_to_smem(in_gmem.at[pl.ds(start_ref[0], 4)], smem, barrier)
      plgpu.barrier_wait(barrier)
      smem[...] = smem[...] + 10
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem, out_gmem.at[pl.ds(start_ref[1], 4)])
      plgpu.wait_smem_to_gmem(0)

    np.testing.assert_array_equal(
        jax.jit(_kernel)(jnp.array([1, 0], jnp.int32), x),
        np.array([[12, 13], [14, 15], [10, 10]], np.int32))

  def test_functional_interpreter_loads_through_tma_before_gmem_as_the_hardware(self):
    # A TMA load whose window starts before its GMEM source reads zeros there,
    # never the source's last rows.
    x = jnp.arange(6, dtype=jnp.int32).reshape((3, 2))

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((3, 2), jnp.int32),
        interpret=InterpretParams(functional=True),
        scratch_types=dict(barrier=plgpu.Barrier(), smem=plgpu.SMEM((4, 2), jnp.int32)),
    )
    def _kernel(start_ref, in_gmem, out_gmem, barrier, smem):
      plgpu.copy_gmem_to_smem(in_gmem.at[pl.ds(start_ref[0], 4)], smem, barrier)
      plgpu.barrier_wait(barrier)
      smem[...] = smem[...] + 10
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem, out_gmem.at[pl.ds(start_ref[1], 4)])
      plgpu.wait_smem_to_gmem(0)

    np.testing.assert_array_equal(
        jax.jit(_kernel)(jnp.array([-1, 0], jnp.int32), x),
        np.array([[10, 10], [10, 11], [12, 13]], np.int32))

  def test_functional_interpreter_stores_through_tma_before_gmem_as_the_hardware(self):
    # A TMA store whose window starts before its GMEM destination drops what
    # lies there, never writing the destination's last rows.
    x = jnp.arange(8, dtype=jnp.float32).reshape((4, 2))

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((6, 2), jnp.float32),
        interpret=InterpretParams(functional=True),
        scratch_types=dict(barrier=plgpu.Barrier(), smem=plgpu.SMEM((4, 2), jnp.float32)),
    )
    def _kernel(start_ref, in_gmem, out_gmem, barrier, smem):
      plgpu.copy_gmem_to_smem(in_gmem, smem, barrier)
      plgpu.barrier_wait(barrier)
      plgpu.copy_smem_to_gmem(smem, out_gmem.at[pl.ds(start_ref[0], 4)])
      plgpu.wait_smem_to_gmem(0)

    out = np.asarray(jax.jit(_kernel)(jnp.array([-1], jnp.int32), x))
    np.testing.assert_array_equal(out[:3], np.asarray(x)[1:])
    self.assertTrue(np.isnan(out[3:]).all())

  @jtu.parameterized.parameters(True, False)
  def test_functional_interpreter_copies_gmem_to_smem_only_where_its_predicate_holds(
      self, predicate):
    # A disabled copy touches no memory, even where its window would lie past
    # its SMEM destination.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((4,), jnp.int32),
        interpret=InterpretParams(functional=True),
        scratch_types=dict(barrier=plgpu.Barrier(), smem=plgpu.SMEM((8,), jnp.int32)),
    )
    def _kernel(flag_ref, start_ref, x_ref, o_ref, barrier, smem):
      smem[...] = jnp.full((8,), -1, jnp.int32)
      plgpu.commit_smem()
      plgpu.copy_gmem_to_smem(x_ref, smem.at[pl.ds(start_ref[0], 4)], barrier,
                              predicate=flag_ref[0] == 1)
      plgpu.barrier_wait(barrier)
      o_ref[...] = smem[pl.ds(0, 4)]

    x = jnp.arange(4, dtype=jnp.int32)
    np.testing.assert_array_equal(
        jax.jit(_kernel)(jnp.array([int(predicate)], jnp.int32),
                         jnp.array([0 if predicate else 6], jnp.int32), x),
        x if predicate else np.full((4,), -1, np.int32))

  def test_functional_interpreter_keeps_a_while_conditions_writes(self):
    # The condition runs before each step and once more to end the loop: each
    # run's write to SMEM stays.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((1,), jnp.int32),
        interpret=InterpretParams(functional=True),
        scratch_types=dict(count=plgpu.SMEM((1,), jnp.int32)),
    )
    def _kernel(o_ref, count):
      count[0] = 0

      def keep_going(i):
        count[0] = count[0] + 1
        return i < 3

      jax.lax.while_loop(keep_going, lambda i: i + 1, 0)
      o_ref[0] = count[0]

    np.testing.assert_array_equal(jax.jit(_kernel)(), np.array([4], np.int32))

  def test_functional_interpreter_raises_on_a_while_conditions_out_of_bounds_read(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((1,), jnp.int32),
        interpret=InterpretParams(functional=True),
    )
    def _kernel(x_ref, n_ref, o_ref):
      def keep_going(i):
        return jnp.sum(x_ref[pl.ds(n_ref[0], 8)]) + i < 0

      o_ref[0] = jax.lax.while_loop(keep_going, lambda i: i + 1, 0)

    with self.assertRaisesRegex(Exception, 'Out-of-bounds'):
      jax.block_until_ready(jax.jit(_kernel)(
          jnp.zeros((8,), jnp.int32), jnp.array([4], jnp.int32)))

  def test_functional_interpreter_refuses_a_block_of_two_threads(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2,), jnp.int32),
        num_threads=2,
        thread_name='x',
        interpret=InterpretParams(functional=True),
    )
    def _kernel(o_ref):
      o_ref[jax.lax.axis_index('x')] = 1

    with self.assertRaisesRegex(NotImplementedError, '2 threads a block'):
      jax.jit(_kernel)()

  def test_load(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 64), jnp.float32),
        scratch_types=dict(smem=plgpu.SMEM((128, 64), jnp.float32)),
        interpret=InterpretParams(),
    )
    def _kernel(x_ref, o_ref, smem):
      smem[...] = plgpu.load(x_ref)
      o_ref[...] = plgpu.load(smem) * 2

    x = jnp.arange(128 * 64, dtype=jnp.float32).reshape(128, 64)
    np.testing.assert_array_equal(_kernel(x), x * 2)

  def test_load_participates_in_race_detection(self):
    def _kernel(x_ref, o_ref, smem):
      thread_idx = jax.lax.axis_index('t')

      @pl.when(thread_idx == 0)
      def _():
        smem[...] = x_ref[...]

      @pl.when(thread_idx == 1)
      def _():
        o_ref[...] = plgpu.load(smem)

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct((8,), jnp.int32),
        scratch_types=dict(smem=plgpu.SMEM((8,), jnp.int32)),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )

    with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
      kernel(jnp.arange(8, dtype=jnp.int32)).block_until_ready()

  def test_store(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 64), jnp.float32),
        scratch_types=dict(smem=plgpu.SMEM((128, 64), jnp.float32)),
        interpret=InterpretParams(),
    )
    def _kernel(x_ref, o_ref, smem):
      plgpu.store(smem, x_ref[...])
      plgpu.store(o_ref, smem[...] * 2, optimized=False)
      plgpu.store(o_ref.at[:16, 8:24], jnp.zeros((16, 16), jnp.float32))

    x = jnp.arange(128 * 64, dtype=jnp.float32).reshape(128, 64)
    np.testing.assert_array_equal(_kernel(x), (x * 2).at[:16, 8:24].set(0))

  @jtu.parameterized.parameters(False, True)
  def test_store_participates_in_race_detection(self, with_race):
    def _kernel(x_ref, o_ref, smem, barrier):
      thread_idx = jax.lax.axis_index('t')

      @pl.when(thread_idx == 0)
      def _():
        plgpu.store(smem, x_ref[...])
        if not with_race:
          plgpu.barrier_arrive(barrier)

      @pl.when(thread_idx == 1)
      def _():
        if not with_race:
          plgpu.barrier_wait(barrier)
        o_ref[...] = smem[...]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct((8,), jnp.int32),
        scratch_types=dict(
            smem=plgpu.SMEM((8,), jnp.int32),
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )

    x = jnp.arange(8, dtype=jnp.int32)
    if with_race:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
        kernel(x).block_until_ready()
    else:
      np.testing.assert_array_equal(kernel(x), x)

  def test_ref_union_disjoint_group_lifetimes_are_allowed(self):
    # Each group is written before it is read, so their lifetimes do not
    # overlap and every read sees what its own group wrote.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_types=dict(
            union=plgpu.RefUnion(
                plgpu.SMEM((128,), jnp.float32),
                plgpu.SMEM((128,), jnp.float32),
            ),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(o_ref, union):
      first, second = union
      first[...] = jnp.arange(128, dtype=jnp.float32)
      x = first[...]
      second[...] = x + 1.0
      o_ref[...] = second[...]

    np.testing.assert_array_equal(
        jax.jit(_kernel)(), np.arange(128, dtype=np.float32) + 1.0
    )

  def test_ref_union_disjoint_members_do_not_race(self):
    # Two refs in the *same* alias group are laid out disjointly, so writing
    # both concurrently is not a race.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_types=dict(
            union=plgpu.RefUnion(
                (plgpu.SMEM((128,), jnp.float32),
                 plgpu.SMEM((128,), jnp.float32)),
            ),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(o_ref, union):
      a, b = union[0]
      tid = jax.lax.axis_index('t')

      @pl.when(tid == 0)
      def _():
        a[...] = jnp.zeros((128,), jnp.float32)
        o_ref[...] = a[...]

      @pl.when(tid == 1)
      def _():
        b[...] = jnp.ones((128,), jnp.float32)

    jax.jit(_kernel)()

  def test_ref_union_member_transforms_are_logical(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 64), jnp.float16),
        scratch_types=dict(
            union=plgpu.RefUnion(
                plgpu.SMEM(
                    (128, 64),
                    jnp.float16,
                    transforms=(
                        plgpu.TilingTransform((8, 64)),
                        plgpu.SwizzleTransform(128),
                    ),
                ),
                plgpu.SMEM((256,), jnp.float32),
            ),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(o_ref, union):
      tiled, _ = union
      tiled[...] = jnp.full((128, 64), 3.0, jnp.float16)
      o_ref[...] = tiled[...]

    np.testing.assert_array_equal(
        jax.jit(_kernel)(), np.full((128, 64), 3.0, np.float16)
    )

  def test_ref_union_member_indexing(self):
    # Indexing a member is ordinary indexing of the member's own buffer.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((32,), jnp.float32),
        scratch_types=dict(
            union=plgpu.RefUnion(
                plgpu.SMEM((4, 32), jnp.float32),
                plgpu.SMEM((128,), jnp.float32),
            ),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(o_ref, union):
      a, _ = union
      a[...] = jnp.arange(128, dtype=jnp.float32).reshape(4, 32)
      o_ref[...] = a[2]

    np.testing.assert_array_equal(
        jax.jit(_kernel)(), np.arange(64, 96, dtype=np.float32)
    )

  def test_ref_union_tmem_members_round_trip(self):
    # Members are allocated at their logical `(rows, cols)` shape and dtype;
    # nothing sees the grid of 32-bit cells they occupy column ranges of, so a
    # packed `bfloat16` member is just another member dtype.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 64), jnp.float32),
        scratch_types=dict(
            union=plgpu.RefUnion(
                plgpu.TMEM((128, 64), jnp.float32),
                (plgpu.TMEM((128, 128), jnp.bfloat16, packed=True),
                 plgpu.TMEM((128, 64), jnp.float32)),
            ),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(o_ref, union):
      first, (packed, second) = union
      plgpu.async_store_tmem(first, jnp.ones((128, 64), jnp.float32))
      plgpu.commit_tmem()
      o_ref[...] = plgpu.async_load_tmem(first)
      plgpu.wait_load_tmem()
      # The other group takes over the storage once `first` is dead.
      plgpu.async_store_tmem(packed, jnp.zeros((128, 128), jnp.bfloat16))
      plgpu.async_store_tmem(second, jnp.zeros((128, 64), jnp.float32))
      plgpu.commit_tmem()

    np.testing.assert_array_equal(
        jax.jit(_kernel)(), np.ones((128, 64), np.float32)
    )

  def test_ref_union_member_as_mma_accumulator(self):
    # The product is formed at the member's dtype, not at the placeholder dtype
    # the union's own aval carries.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 128), jnp.float32),
        scratch_types=dict(
            a_smem=plgpu.SMEM((128, 128), jnp.float16),
            b_smem=plgpu.SMEM((128, 128), jnp.float16),
            union=plgpu.RefUnion(
                plgpu.TMEM((128, 128), jnp.float32),
                plgpu.TMEM((128, 128), jnp.float32),
            ),
            barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(a, b, out_ref, a_smem, b_smem, union, barrier):
      acc_ref, _ = union
      a_smem[...] = a[...]
      b_smem[...] = b[...]
      plgpu.commit_smem()
      plgpu.tcgen05_mma(acc_ref, a_smem, b_smem, accumulate=False)
      plgpu.tcgen05_commit_arrive(barrier, predicate=False)
      plgpu.tcgen05_mma(acc_ref, a_smem, b_smem, accumulate=True)
      plgpu.tcgen05_commit_arrive(barrier, predicate=True)
      plgpu.barrier_wait(barrier)
      out_ref[...] = plgpu.async_load_tmem(acc_ref)

    a = jax.random.uniform(jax.random.key(0), (128, 128), jnp.float16)
    b = jax.random.uniform(jax.random.key(1), (128, 128), jnp.float16)
    expected = 2 * np.matmul(np.asarray(a), np.asarray(b), dtype=np.float32)
    self.assertArraysEqual(_kernel(a, b), expected)

  def test_block_scaled_mma_multiplies_each_block_by_its_scales(self):
    # Each operand's blocks of 32 along K times their E8M0 scales, copied to
    # TMEM from the tiles `tcgen05.scales_layout` lays them out in.
    m = n = 128
    k, block = 256, 32
    scales = k // block

    def tiled(values):
      # `values` (MN, K / 32) as `async_copy_scales_to_tmem` reads them.
      mn, kk = values.shape
      return (
          values.reshape(mn // 128, 4, 32, kk // 4, 4)
          .transpose(0, 3, 2, 1, 4)
          .reshape(mn // 128, kk // 4, 32, 16)
      )

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((m, n), jnp.float32),
        scratch_types=dict(
            a_smem=plgpu.SMEM((m, k), jnp.float8_e4m3fn),
            b_smem=plgpu.SMEM((n, k), jnp.float8_e4m3fn),
            a_scale_smem=plgpu.SMEM((m // 128, scales // 4, 32, 16),
                                    jnp.float8_e8m0fnu),
            b_scale_smem=plgpu.SMEM((n // 128, scales // 4, 32, 16),
                                    jnp.float8_e8m0fnu),
            acc=plgpu.TMEM((m, n), jnp.float32),
            a_scale=plgpu.TMEM((m, scales), jnp.float8_e8m0fnu,
                               layout=plgpu.TMEMLayout.SCALES_LAYOUT),
            b_scale=plgpu.TMEM((n, scales), jnp.float8_e8m0fnu,
                               layout=plgpu.TMEMLayout.SCALES_LAYOUT),
            barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(a, b, a_scales, b_scales, out_ref, a_smem, b_smem,
                a_scale_smem, b_scale_smem, acc, a_scale, b_scale, barrier):
      a_smem[...] = a[...]
      b_smem[...] = b[...]
      a_scale_smem[...] = a_scales[...]
      b_scale_smem[...] = b_scales[...]
      plgpu.commit_smem()
      plgpu.async_copy_scales_to_tmem(a_scale_smem, a_scale)
      plgpu.async_copy_scales_to_tmem(b_scale_smem, b_scale)
      plgpu.tcgen05_mma(acc, a_smem, b_smem.transpose((1, 0)),
                        a_scale=a_scale, b_scale=b_scale, accumulate=False)
      plgpu.tcgen05_commit_arrive(barrier)
      plgpu.barrier_wait(barrier)
      out_ref[...] = plgpu.async_load_tmem(acc)

    a = jax.random.normal(jax.random.key(0), (m, k)).astype(jnp.float8_e4m3fn)
    b = jax.random.normal(jax.random.key(1), (n, k)).astype(jnp.float8_e4m3fn)
    a_scales = jax.lax.bitcast_convert_type(
        jax.random.randint(jax.random.key(2), (m, scales), 122, 132,
                           dtype=jnp.uint8), jnp.float8_e8m0fnu)
    b_scales = jax.lax.bitcast_convert_type(
        jax.random.randint(jax.random.key(3), (n, scales), 122, 132,
                           dtype=jnp.uint8), jnp.float8_e8m0fnu)
    result = _kernel(a, b, tiled(a_scales), tiled(b_scales))

    def dequantized(values, scales):
      return np.asarray(values, np.float32) * np.repeat(
          np.asarray(scales, np.float32), block, axis=1)

    expected = np.matmul(dequantized(a, a_scales), dequantized(b, b_scales).T)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6)

  def test_transpose_ref(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2, 64, 128), jnp.float32),
        scratch_types=dict(smem=plgpu.SMEM((128, 64), jnp.float32)),
        interpret=InterpretParams(),
    )
    def _kernel(x_ref, o_ref, smem):
      # Store through the transposed view, then load back through it and
      # through the plain ref.
      smem.T[...] = x_ref[...]
      o_ref[0] = smem.T[...]
      o_ref[1] = smem[...].T

    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    y = _kernel(x)
    np.testing.assert_array_equal(y[0], x)
    np.testing.assert_array_equal(y[1], x)

  def test_indexing_a_transposed_ref(self):
    # The indexer applies to the transposed view, not to the buffer.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_types=dict(smem=plgpu.SMEM((128, 64), jnp.float32)),
        interpret=InterpretParams(),
    )
    def _kernel(x_ref, o_ref, smem):
      smem[...] = x_ref[...]
      o_ref[...] = smem.T[3]

    x = jnp.arange(128 * 64, dtype=jnp.float32).reshape(128, 64)
    np.testing.assert_array_equal(_kernel(x), np.asarray(x).T[3])

  def test_transposed_slice_of_a_tiled_ref(self):
    # Based on tokamax pattern
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 64), jnp.float16),
        scratch_types=dict(
            smem=plgpu.SMEM(
                (2, 128, 64),
                jnp.float16,
                transforms=(
                    plgpu.TilingTransform((8, 64)),
                    plgpu.SwizzleTransform(128),
                ),
            )
        ),
        interpret=InterpretParams(),
    )
    def _kernel(x_ref, o_ref, smem):
      smem.at[1].T[...] = x_ref[...]
      o_ref[...] = smem.at[1][...]

    x = jnp.arange(64 * 128, dtype=jnp.float16).reshape(64, 128)
    np.testing.assert_array_equal(_kernel(x), np.asarray(x).T)

  def test_mma_with_a_transposed_ref_operand(self):
    # Based on tokamax pattern
    def tiled_smem(shape):
      return plgpu.SMEM(
          shape,
          jnp.float16,
          transforms=(
              plgpu.TilingTransform((8, 64)),
              plgpu.SwizzleTransform(128),
          ),
      )

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 128), jnp.float32),
        scratch_types=dict(
            w_smem=tiled_smem((128, 128)),
            x_smem=tiled_smem((2, 128, 128)),
            acc_tmem=plgpu.TMEM((128, 128), jnp.float32),
            done=plgpu.Barrier(num_arrivals=1, orders_tensor_core=True),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(w_ref, x_ref, o_ref, w_smem, x_smem, acc_tmem, done):
      w_smem[...] = w_ref[...]
      x_smem.at[1][...] = x_ref[...]
      plgpu.commit_smem()
      plgpu.tcgen05_mma(acc_tmem, w_smem, x_smem.at[1].T, accumulate=False)
      plgpu.tcgen05_commit_arrive(done)
      plgpu.barrier_wait(done)
      o_ref[...] = plgpu.async_load_tmem(acc_tmem)
      plgpu.wait_load_tmem()

    w = (jnp.arange(128 * 128, dtype=jnp.float16).reshape(128, 128) % 7) - 3
    x = (jnp.arange(128 * 128, dtype=jnp.float16).reshape(128, 128) % 5) - 2
    np.testing.assert_allclose(
        _kernel(w, x),
        np.asarray(w, np.float32) @ np.asarray(x, np.float32).T,
        atol=1e-3,
    )

  @jtu.parameterized.named_parameters(
      ('raise', 'raise'), ('uninitialized', 'uninitialized')
  )
  def test_out_of_bounds_dynamic_read(self, out_of_bounds_reads):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((64,), jnp.float32),
        scratch_types=dict(smem=plgpu.SMEM((128,), jnp.float32)),
        interpret=InterpretParams(
            out_of_bounds_reads=out_of_bounds_reads,
            uninitialized_memory='nan',
        ),
    )
    def _kernel(start_ref, o_ref, smem):
      smem[...] = jnp.arange(128, dtype=jnp.float32)
      o_ref[...] = smem[pl.ds(start_ref[0], 64)]

    start = jnp.array([100], jnp.int32)
    if out_of_bounds_reads == 'raise':
      with self.assertRaisesRegex(Exception, r'Out-of-bounds read'):
        _kernel(start)
      return
    y = np.asarray(_kernel(start))
    np.testing.assert_array_equal(y[:28], np.arange(100, 128, dtype=np.float32))
    self.assertTrue(np.all(np.isnan(y[28:])))

  def test_out_of_bounds_dynamic_write(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_types=dict(smem=plgpu.SMEM((128,), jnp.float32)),
        interpret=InterpretParams(),
    )
    def _kernel(start_ref, o_ref, smem):
      smem[...] = jnp.zeros((128,), jnp.float32)
      smem[pl.ds(start_ref[0], 64)] = jnp.ones((64,), jnp.float32)
      o_ref[...] = smem[...]

    with self.assertRaisesRegex(Exception, r'Out-of-bounds swap'):
      _kernel(jnp.array([100], jnp.int32))

  @jtu.parameterized.named_parameters(('partially', 3), ('entirely', 4))
  def test_copy_smem_to_gmem_out_of_bounds(self, start):
    # As on hardware, a TMA store writes the part of its window that is inside
    # the destination and drops the rest.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((4, 4), jnp.int32),
        interpret=InterpretParams(uninitialized_memory='zero'),
        scratch_types=dict(smem_ref=plgpu.SMEM((2, 4), jnp.int32)),
    )
    def _kernel(start_ref, out_gmem, smem_ref):
      smem_ref[...] = jnp.arange(8, dtype=jnp.int32).reshape((2, 4))
      # The destination window runs past the end of the output.
      plgpu.copy_smem_to_gmem(smem_ref, out_gmem.at[pl.ds(start_ref[0], 2)])
      plgpu.wait_smem_to_gmem(0)

    y = np.asarray(_kernel(jnp.array([start], jnp.int32)))
    expected = np.zeros((4, 4), np.int32)
    expected[start:] = np.arange(8, dtype=np.int32).reshape((2, 4))[: 4 - start]
    np.testing.assert_array_equal(y, expected)

  def test_copy_gmem_to_smem_out_of_bounds_destination(self):
    # Unlike an overhanging GMEM destination, one in SMEM has nowhere for the
    # rest of the window to go.
    x = jnp.arange(4, dtype=jnp.int32).reshape((2, 2))

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((4, 2), jnp.int32),
        interpret=InterpretParams(),
        scratch_types=dict(
            barrier=plgpu.Barrier(), smem=plgpu.SMEM((4, 2), jnp.int32)
        ),
    )
    def _kernel(start_ref, in_gmem, out_gmem, barrier, smem):
      plgpu.copy_gmem_to_smem(in_gmem, smem.at[pl.ds(start_ref[0], 2)], barrier)
      plgpu.barrier_wait(barrier)
      out_gmem[...] = smem[...]

    with self.assertRaisesRegex(
        Exception, r'Out-of-bounds store of HostAllocationKey\(.+\)'
    ):
      _kernel(jnp.array([3], jnp.int32), x)

  def test_copy_gmem_to_smem_out_of_bounds(self):
    # A TMA load may read past the end of its source; the missing elements are
    # zero-filled by default.
    x = jnp.arange(6, dtype=jnp.int32).reshape((3, 2))

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((4, 2), jnp.int32),
        interpret=InterpretParams(uninitialized_memory='nan'),
        scratch_types=dict(
            barrier=plgpu.Barrier(), smem=plgpu.SMEM((4, 2), jnp.int32)
        ),
    )
    def _kernel(start_ref, in_gmem, out_gmem, barrier, smem):
      plgpu.copy_gmem_to_smem(in_gmem.at[pl.ds(start_ref[0], 4)], smem, barrier)
      plgpu.barrier_wait(barrier)
      out_gmem[...] = smem[...]

    y = np.asarray(_kernel(jnp.array([0], jnp.int32), x))
    np.testing.assert_array_equal(y[:3], x)
    np.testing.assert_array_equal(y[3], np.zeros((2,), np.int32))

  def test_copy_gmem_to_smem_im2col_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((8, 8), jnp.int16),
        interpret=InterpretParams(),
        scratch_types=dict(
            barrier=plgpu.Barrier(), smem=plgpu.SMEM((8, 8), jnp.int16)
        ),
    )
    def _kernel(in_gmem, out_gmem, barrier, smem):
      in_im2col = plgpu.im2col_ref(in_gmem, window_shape=(3,))
      plgpu.copy_gmem_to_smem(in_im2col.at[:, 0, :], smem, barrier)
      plgpu.barrier_wait(barrier)
      out_gmem[...] = smem[...]

    with self.assertRaisesRegex(
        NotImplementedError, 'does not support im2col copies'
    ):
      _kernel(jnp.zeros((1, 10, 8), jnp.int16))

  @jtu.parameterized.product(
      memory_space=['smem', 'tmem'], synchronized=[False, True]
  )
  def test_ref_union_groups_alias(self, memory_space, synchronized):
    if memory_space == 'smem':
      member = plgpu.SMEM((128,), jnp.float32)

      def store(ref, value):
        ref[...] = value

    else:
      member = plgpu.TMEM((128, 64), jnp.float32)

      def store(ref, value):
        plgpu.async_store_tmem(ref, value)
        plgpu.commit_tmem()

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_types=dict(
            union=plgpu.RefUnion(member, member),
            done=plgpu.Barrier(num_arrivals=1),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(o_ref, union, done):
      first, second = union
      tid = jax.lax.axis_index('t')

      @pl.when(tid == 0)
      def _():
        store(first, jnp.ones(member.shape, member.dtype))
        if synchronized:
          plgpu.barrier_arrive(done)

      @pl.when(tid == 1)
      def _():
        if synchronized:
          plgpu.barrier_wait(done)
        store(second, jnp.zeros(member.shape, member.dtype))

    if synchronized:
      jax.jit(_kernel)()
    else:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
        jax.jit(_kernel)().block_until_ready()

  def test_ref_union_read_of_other_group_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_types=dict(
            union=plgpu.RefUnion(
                plgpu.SMEM((128,), jnp.float32),
                plgpu.SMEM((128,), jnp.float32),
            ),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(o_ref, union):
      first, second = union
      first[...] = jnp.arange(128, dtype=jnp.float32)
      o_ref[...] = second[...]

    with self.assertRaisesRegex(
        Exception,
        r'through alias group 1, which was last written through alias group 0',
    ):
      jax.jit(_kernel)()

  def test_tiling_and_swizzle_transforms(self):

    @jax.jit
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 128), jnp.float16),
        scratch_types=dict(
            smem_ref1=plgpu.SMEM(
                (2, 128, 128),
                jnp.float16,
                transforms=(
                    plgpu.TilingTransform((8, 64)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
            smem_ref2=plgpu.SMEM(
                (4, 128, 32),
                jnp.float16,
                transforms=(plgpu.SwizzleTransform(64),),
            ),
            smem_ref3=plgpu.SMEM(
                (256, 128),
                jnp.float16,
                transforms=(plgpu.TilingTransform((16, 32)),),
            ),
        ),
        interpret=InterpretParams(),
    )
    def kernel(o_ref, smem_ref1, smem_ref2, smem_ref3):
      smem_ref1[...] = jnp.full_like(smem_ref1, 42.0)
      smem_ref2[...] = jnp.full_like(smem_ref2, 1.0)
      smem_ref3[...] = jnp.full_like(smem_ref3, 2.0)
      o_ref[...] = (smem_ref1[0, ...] + smem_ref2[...].reshape((128, 128))
                    + smem_ref3[:128])

    np.testing.assert_equal(kernel(), np.full((128, 128), 45.0, jnp.float16))

  def test_tiling_and_swizzle_transforms_with_pallas_call(self):
    def _kernel(o_ref, smem_ref1, smem_ref2, smem_ref3):
      smem_ref1[...] = jnp.full_like(smem_ref1, 42.0)
      smem_ref2[...] = jnp.full_like(smem_ref2, 1.0)
      smem_ref3[...] = jnp.full_like(smem_ref3, 2.0)
      o_ref[...] = (smem_ref1[0, ...] + smem_ref2[...].reshape((128, 128))
                    + smem_ref3[:128])

    @jax.jit
    def run():
      return pl.pallas_call(
          _kernel,
          out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float16),
          out_specs=plgpu.BlockSpec((128, 128), memory_space=plgpu.SMEM),
          scratch_shapes=dict(
              smem_ref1=plgpu.SMEM((2, 128, 128), jnp.float16, transforms=(
                  plgpu.TilingTransform((8, 64)),
                  plgpu.SwizzleTransform(128),
              )),
              smem_ref2=plgpu.SMEM((4, 128, 32), jnp.float16, transforms=(
                  plgpu.SwizzleTransform(64),
              )),
              smem_ref3=plgpu.SMEM((256, 128), jnp.float16, transforms=(
                  plgpu.TilingTransform((16, 32)),
              )),
          ),
          interpret=InterpretParams(),
      )()

    np.testing.assert_equal(run(), np.full((128, 128), 45.0, jnp.float16))

  def test_skip_floating_point_ops(self):
    def matmul_kernel(x_ref, y_ref, z_ref):
      # TODO(nrink): Matrix multiplication with `@` is nor supported for real
      # GPU kernels (but the GPU kernel interpreter allows this). Replace this
      # with a `wgmma` or `tcgen05_mma` once these are supported by the GPU
      # kernel interpreter.
      z_ref[...] = x_ref[...] @ y_ref[...]

    def matmul(x: jax.Array, y: jax.Array):
      return pl.pallas_call(
          matmul_kernel,
          out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
          interpret=InterpretParams(
              skip_floating_point_ops=True
          ),
      )(x, y)

    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (1024, 1024))
    y = jax.random.normal(k2, (1024, 1024))
    z = jax.jit(matmul)(x, y)
    np.testing.assert_array_equal(z, jnp.full_like(z, jnp.inf))

    lowered = jax.jit(matmul).lower(x, y).as_text(dialect='stablehlo')
    self.assertNotIn('dot_general', lowered)

  @jtu.parameterized.parameters(
      (1, 1, 1),
      (2, 1, 2),
      (2, 2, 1),
      (4, 1, 4),
      (4, 2, 2),
      (4, 4, 1),
      (8, 1, 8),
      (8, 2, 4),
      (8, 4, 2),
      (8, 8, 1),
      (16, 1, 16),
      (16, 2, 8),
      (16, 4, 4),
      (16, 8, 2),
      (16, 16, 1),
  )
  def test_matmul_example(self, num_threads, num_row_blocks, num_col_blocks):
    assert num_threads == num_row_blocks * num_col_blocks

    @jax.jit
    def matmul(x: jax.Array, y: jax.Array):
      num_rows_per_block = x.shape[0] // num_row_blocks
      num_cols_per_block = y.shape[1] // num_col_blocks

      @functools.partial(
          plgpu.kernel,
          out_type=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
          num_threads=num_threads,
          thread_name='t',
          interpret=InterpretParams(
              detect_races=True, num_cores_or_threads=num_threads
          ),
      )
      def _matmul_kernel(x_ref, y_ref, o_ref):
        thread_idx = jax.lax.axis_index('t')

        row_block_idx = thread_idx // num_col_blocks
        row_slice = pl.ds(
            row_block_idx * num_rows_per_block, num_rows_per_block
        )

        col_block_idx = jax.lax.rem(thread_idx, jnp.int32(num_col_blocks))
        col_slice = pl.ds(
            col_block_idx * num_cols_per_block, num_cols_per_block
        )

        # TODO(nrink): Matrix multiplication with `@` is nor supported for real
        # GPU kernels (but the GPU kernel interpreter allows this). Replace this
        # with a `wgmma` or `tcgen05_mma` once these are supported by the GPU
        # kernel interpreter.
        o_ref[row_slice, col_slice] = x_ref[row_slice, :] @ y_ref[:, col_slice]

      return _matmul_kernel(x, y)

    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (1024, 1024))
    y = jax.random.normal(k2, (1024, 1024))
    z = matmul(x, y)
    np.testing.assert_allclose(z, x @ y, atol=1e-3)

  def test_run_scoped(self):

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2, 16, 128), jnp.float32),
        num_threads=2,
        thread_name='n',
        interpret=InterpretParams(
            detect_races=True,
        ),
    )
    def f(o_ref):
      def body(ref):
        @pl.when(jax.lax.axis_index('n') == 0)
        def _():
          ref[...] = jnp.zeros_like(ref[...])
          o_ref[0, ...] = ref[...]

        @pl.when(jax.lax.axis_index('n') == 1)
        def _():
          ref[...] = jnp.ones_like(ref[...])
          o_ref[1, ...] = ref[...]

      pl.run_scoped(
          body,
          plgpu.GMEM(o_ref.shape[1:], dtype=o_ref.dtype),
          collective_axes=('n',),
      )

    with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
      f().block_until_ready()

  @jtu.parameterized.parameters(
      ((),),
      (('t',),),
      (('c0',),),
      (('c1',),),
      (('c0', 't'),),
      (('c1', 't'),),
      (('c0', 'c1'),),
      (('c0', 'c1', 't'),),
  )
  def test_run_scoped_with_cluster(self, collective_axes):
    all_axis_names = ('c0', 'c1', 't')
    non_collective_axis_names = tuple(
        name for name in all_axis_names if name not in collective_axes
    )

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2, 2, 2), dtype=jnp.int32),
        cluster=(2, 2),
        cluster_names=('c0', 'c1'),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(
            detect_races=True,
        ),
    )
    def f(o_ref):
      def body(ref):
        collective_indices = tuple(
            jax.lax.axis_index(axis_name) for axis_name in collective_axes
        )
        non_collective_indices = tuple(
            jax.lax.axis_index(axis_name)
            for axis_name in non_collective_axis_names
        )

        ref[collective_indices] = sum(
            stride * index
            for stride, index in zip((1, 2, 4), reversed(collective_indices))
        )

        o_ref[non_collective_indices + collective_indices] = ref[
            collective_indices
        ]

      pl.run_scoped(
          body,
          plgpu.GMEM((2,) * len(collective_axes), dtype=jnp.int32),
          collective_axes=collective_axes,
      )

    if 'c0' in collective_axes or 'c1' in collective_axes:
      with self.assertRaisesRegex(
          Exception,
          r'Collective allocations along cluster axes are not' r' supported\.',
      ):
        _ = f()
    elif 't' not in collective_axes:
      with self.assertRaisesRegex(
          Exception,
          r'Scoped allocation must have the thread axis in its collective'
          r' axes\.',
      ):
        _ = f()
    else:
      y = f()
      expected = np.arange(2 ** len(collective_axes)).reshape(
          (1,) * len(non_collective_axis_names) + (2,) * len(collective_axes)
      )
      expected = np.broadcast_to(expected, (2, 2, 2))
      np.testing.assert_array_equal(y, expected)

  def test_run_scoped_with_unknown_collective_axis(self):

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), dtype=jnp.int32),
        cluster=(2, 2),
        cluster_names=('c0', 'c1'),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(),
    )
    def f(o_ref):
      def body(_):
        o_ref[...] = 42

      pl.run_scoped(
          body,
          plgpu.GMEM((), dtype=jnp.int32),
          collective_axes=('unknown_axis',),
      )

    with self.assertRaisesRegex(
        Exception,
        r"Collective axis `unknown_axis` not found among axes `\['c0', 'c1', 't'\]`",
    ):
      _ = f()

  @jtu.parameterized.parameters(
      (
          True,
          ('c0',),
          r'Collective allocations along cluster axes are not'
          r' supported\.',
      ),
      (
          True,
          ('c0', 't'),
          r'Collective allocations along cluster axes are not'
          r' supported\.',
      ),
      (
          True,
          (),
          (
              r'Scoped allocation must have the thread axis in its collective'
              r' axes\.'
          ),
      ),
      (
          False,
          ('c0',),
          r'Requesting collective allocations, but no explicit thread axis'
          r' specified\.',
      ),
      (False, ('t',), r'Collective axis `t` not found among axes'),
  )
  def test_run_scoped_barrier_with_incorrect_collective_axes(
      self, has_thread_axis, collective_axes, expected_error_regex
  ):
    mesh_kwargs = dict(
        cluster=(2, 2),
        cluster_names=('c0', 'c1'),
    )
    if has_thread_axis:
      mesh_kwargs.update(
          num_threads=2,
          thread_name='t',
      )

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), dtype=jnp.int32),
        interpret=InterpretParams(),
        **mesh_kwargs,
    )
    def f(o_ref):
      def body(_):
        o_ref[...] = 42

      pl.run_scoped(
          body,
          plgpu.Barrier(),
          collective_axes=collective_axes,
      )

    with self.assertRaisesRegex(Exception, expected_error_regex):
      _ = f()

  # Test adapted from
  # https://docs.jax.dev/en/latest/pallas/gpu/reference.html#using-multiple-pallas-threads-per-cuda-block
  def test_producer_consumer_threads_with_barrier(self):
    x = jnp.arange(128, dtype=jnp.float32)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            smem_ref=plgpu.SMEM(x.shape, x.dtype),
            barrier_ref=plgpu.Barrier(),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(x_ref, out_ref, smem_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def producer_thread():
        smem_ref[...] = x_ref[...] + 1
        plgpu.barrier_arrive(barrier_ref)

      @pl.when(thread_id == 1)
      def consumer_thread():
        plgpu.barrier_wait(barrier_ref)
        out_ref[...] = smem_ref[...] + 1

    y = _kernel(x)
    np.testing.assert_array_equal(y, x + 2)

  @jtu.parameterized.parameters(1, 2)
  def test_consumers_of_alternate_tiles_of_one_ring_skip_its_phases(
      self, rotations):
    # A two-slot ring, each tile `rotations` rotations of its steps, whose tiles
    # two consumers take in turn: each waits at a slot's phases of its own
    # tiles, naming each (a ping-pong over one ring), and lets its partner's
    # pass, two or more a slot where a tile takes two rotations; the producer
    # waits for each slot's release at the phase it names. A tile's consumer
    # starts once the tile before it has had its steps (`order`, a phase a
    # tile, each consumer waiting at every other): no consumer names a phase
    # two ahead.
    stages, tiles = 2, 4
    steps = stages * rotations
    turns = tiles * steps
    x = jnp.arange(turns * 128, dtype=jnp.float32).reshape(turns, 128)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            smem=plgpu.SMEM((stages, 128), x.dtype),
            full=plgpu.Barrier(num_barriers=stages),
            consumed=plgpu.Barrier(num_barriers=stages),
            order=plgpu.Barrier(),
        ),
        num_threads=3,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(x_ref, out_ref, smem, full, consumed, order):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def producer():
        @pl.loop(0, turns)
        def _(turn):
          slot, rotation = turn % stages, turn // stages
          pl.when(rotation > 0)(lambda: plgpu.barrier_wait(
              consumed.at[slot], phase=rotation - 1))
          smem[slot] = x_ref[turn] + 1
          plgpu.barrier_arrive(full.at[slot])

      def consumer(first):
        @pl.loop(first, tiles, step=2)
        def _(tile):
          pl.when(tile > 0)(
              lambda: plgpu.barrier_wait(order, phase=tile - 1))
          for step in range(steps):
            turn, slot = tile * steps + step, step % stages
            plgpu.barrier_wait(full.at[slot], phase=turn // stages)
            out_ref[turn] = smem[slot] + 1
            plgpu.barrier_arrive(consumed.at[slot])
          plgpu.barrier_arrive(order)

      pl.when(thread_id == 1)(lambda: consumer(0))
      pl.when(thread_id == 2)(lambda: consumer(1))

    np.testing.assert_array_equal(_kernel(x), x + 2)

  def test_consumer_warps_of_alternate_tiles_of_one_ring_skip_its_phases(self):
    # The ping-pong over one ring with its consumers the four warps of one
    # warpgroup, each the tiles of its index modulo four in turn (a turn
    # barrier a warp, its predecessor arriving once its tile is done: one
    # order barrier would have a warp name a phase two ahead before its
    # first tile): three tiles pass a warp by between its own, so a slot's
    # phases pass the warpgroup its warps' observations give it as well as
    # each warp.
    stages, tiles, warps = 2, 8, 4
    turns = tiles * stages
    x = jnp.arange(turns * 128, dtype=jnp.float32).reshape(turns, 128)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            smem=plgpu.SMEM((stages, 128), x.dtype),
            full=plgpu.Barrier(num_barriers=stages),
            consumed=plgpu.Barrier(num_barriers=stages),
            ordered=plgpu.Barrier(num_barriers=warps),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(x_ref, out_ref, smem, full, consumed, ordered):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def producer():
        @pl.loop(0, turns)
        def _(turn):
          slot, rotation = turn % stages, turn // stages
          pl.when(rotation > 0)(lambda: plgpu.barrier_wait(
              consumed.at[slot], phase=rotation - 1))
          smem[slot] = x_ref[turn] + 1
          plgpu.barrier_arrive(full.at[slot])

      @pl.when(thread_id == 1)
      def consumers():
        @plgpu.warp_map
        def _(warp_id):
          @pl.loop(warp_id, tiles, step=warps)
          def _(tile):
            # Warp 0's j-th tile waits its predecessor's (j - 1)-th arrival,
            # every other warp's its j-th.
            pl.when(tile > 0)(lambda: plgpu.barrier_wait(
                ordered.at[warp_id],
                phase=tile // warps - (warp_id == 0).astype(jnp.int32)))
            for step in range(stages):
              turn = tile * stages + step
              plgpu.barrier_wait(full.at[step], phase=turn // stages)
              out_ref[turn] = smem[step] + 1
              plgpu.barrier_arrive(consumed.at[step])
            plgpu.barrier_arrive(ordered.at[(warp_id + 1) % warps])

    np.testing.assert_array_equal(_kernel(x), x + 2)

  def test_a_named_wait_woken_past_its_phase_is_refused(self):
    # A waiter blocked at the phase the barrier is completing, which completes
    # it and two more before the waiter runs again (they arrive while the
    # barrier's lock is held, the waiter inside its wait): the hardware's poll
    # at its phase's parity would see none of them, so its wait is refused.
    from jax._src.pallas.mosaic_gpu.interpret import shared_memory

    memory = shared_memory.GPUSharedMemory(
        num_devices=1, out_of_bounds_reads="raise", dma_execution_mode="eager",
        uninitialized_memory="nan", detect_races=False,
        barrier=threading.Barrier(1), clean_up_barrier=threading.Barrier(1),
        num_threads_per_block=2, num_blocks_per_cluster=1,
        num_tma_threads_per_device=1)
    barrier = shared_memory.Barrier(
        memory, num_pallas_threads_per_block=2, ref_count=1, num_arrivals=1,
        orders_tensor_core=False)
    waiter = shared_memory.Warpgroup(0, 0, 0, 0)
    barrier.arrive(None)
    barrier.wait(waiter, phase=0)

    entered, refused = threading.Event(), []
    blocked = barrier.cv.wait

    def waiting(*args, **kwargs):
      entered.set()
      return blocked(*args, **kwargs)

    barrier.cv.wait = waiting

    def wait():
      try:
        barrier.wait(waiter, phase=1)
      except ValueError as error:
        refused.append(error)

    thread = threading.Thread(target=wait)
    thread.start()
    entered.wait()
    with barrier.cv:
      for _ in range(3):
        barrier.arrive(None)
    thread.join()
    self.assertLen(refused, 1)
    self.assertIn("the hardware's wait at that parity", str(refused[0]))

  @jtu.parameterized.parameters(
      ("behind", 3, 0), ("ahead", 1, 2), ("latest", 3, 2), ("pending", 0, 0))
  def test_a_wait_names_the_latest_or_the_pending_phase(
      self, name, observed, phase):
    # A thread observes `observed` phases, each as it completes, then waits at
    # `phase`: two behind the latest completion, or past the one the barrier is
    # completing, is a phase the hardware's parity wait takes another for; the
    # latest and the pending phase are waited at. The other thread completes
    # the pending phase.
    del name
    refused = not observed - 1 <= phase <= observed

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_types=dict(barrier=plgpu.Barrier()),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        for completed in range(observed):
          plgpu.barrier_arrive(barrier)
          plgpu.barrier_wait(barrier, phase=completed)
        plgpu.barrier_wait(barrier, phase=phase)
        out_ref[...] = jnp.ones_like(out_ref)

      @pl.when(thread_id == 1)
      def _():
        if phase == observed:
          plgpu.barrier_arrive(barrier)

    if refused:
      with self.assertRaisesRegex(
          jax.errors.JaxRuntimeError, "the hardware's wait at that parity"):
        _kernel().block_until_ready()
    else:
      np.testing.assert_array_equal(_kernel(), np.ones((128,), np.float32))

  @jtu.parameterized.product(with_race=[True, False])
  def test_barrier_multidimensional_1d(self, with_race):
    shape = (2,)
    x = jnp.arange(2, dtype=jnp.float32)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            smem_ref=plgpu.SMEM(shape, x.dtype),
            barrier=plgpu.Barrier(num_arrivals=1, num_barriers=shape),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(x_ref, out_ref, smem_ref, barrier):
      thread_id = jax.lax.axis_index('t')
      for i in range(2):
        @pl.when(thread_id == 0)
        def _():
          smem_ref[i] = x_ref[i] + 1
          plgpu.barrier_arrive(barrier.at[i])

        @pl.when(thread_id == 1)
        def _():
          if not with_race:
            plgpu.barrier_wait(barrier.at[i])
          out_ref[i] = smem_ref[i] + 1

    if with_race:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
        _kernel(x).block_until_ready()
    else:
      y = _kernel(x)
      np.testing.assert_array_equal(y, x + 2)

  @jtu.parameterized.product(with_race=[True, False])
  def test_barrier_multidimensional_2d(self, with_race):
    shape = (2, 3)
    x = jnp.arange(6, dtype=jnp.float32).reshape(shape)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            smem_ref=plgpu.SMEM(shape, x.dtype),
            barrier=plgpu.Barrier(num_arrivals=1, num_barriers=shape),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(x_ref, out_ref, smem_ref, barrier):
      thread_id = jax.lax.axis_index('t')
      for i in range(2):
        for j in range(3):
          @pl.when(thread_id == 0)
          def _():
            smem_ref[i, j] = x_ref[i, j] + 1
            plgpu.barrier_arrive(barrier.at[i, j])

          @pl.when(thread_id == 1)
          def _():
            if not with_race:
              plgpu.barrier_wait(barrier.at[i, j])
            out_ref[i, j] = smem_ref[i, j] + 1

    if with_race:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
        _kernel(x).block_until_ready()
    else:
      y = _kernel(x)
      np.testing.assert_array_equal(y, x + 2)

  @jtu.parameterized.product(with_race=[True, False])
  def test_barrier_multidimensional_3d(self, with_race):
    shape = (2, 1, 3)
    x = jnp.arange(6, dtype=jnp.float32).reshape(shape)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            smem_ref=plgpu.SMEM(shape, x.dtype),
            barrier=plgpu.Barrier(num_arrivals=1, num_barriers=shape),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(x_ref, out_ref, smem_ref, barrier):
      thread_id = jax.lax.axis_index('t')
      for i in range(2):
        for j in range(1):
          for k in range(3):
            @pl.when(thread_id == 0)
            def _():
              smem_ref[i, j, k] = x_ref[i, j, k] + 1
              plgpu.barrier_arrive(barrier.at[i, j, k])

            @pl.when(thread_id == 1)
            def _():
              if not with_race:
                plgpu.barrier_wait(barrier.at[i, j, k])
              out_ref[i, j, k] = smem_ref[i, j, k] + 1

    if with_race:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
        _kernel(x).block_until_ready()
    else:
      y = _kernel(x)
      np.testing.assert_array_equal(y, x + 2)

  @jtu.parameterized.parameters(range(2, 17))
  def test_single_barrier_with_multiple_arrival(self, num_threads):

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((num_threads - 1,), jnp.int32),
            barrier_ref=plgpu.Barrier(
                num_arrivals=num_threads - 1, num_barriers=1
            ),
        ),
        num_threads=num_threads,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, smem_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        plgpu.barrier_wait(barrier_ref)
        out_ref[...] = sum(smem_ref[...])

      @pl.when(thread_id > 0)
      def _():
        smem_ref[thread_id - 1] = thread_id
        plgpu.barrier_arrive(barrier_ref)

    y = _kernel()
    self.assertEqual(y, sum(range(num_threads)))

  @jtu.parameterized.parameters(range(2, 17))
  def test_multiple_barriers_with_single_arrival(self, num_threads):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((num_threads - 1,), jnp.int32),
            barrier_ref=plgpu.Barrier(
                num_arrivals=1, num_barriers=num_threads - 1
            ),
        ),
        num_threads=num_threads,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, smem_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        for b in range(num_threads - 1):
          plgpu.barrier_wait(barrier_ref.at[b])
        out_ref[...] = sum(smem_ref[...])

      @pl.when(thread_id > 0)
      def _():
        smem_ref[thread_id - 1] = thread_id
        plgpu.barrier_arrive(barrier_ref.at[thread_id - 1])

    y = _kernel()
    self.assertEqual(y, sum(range(num_threads)))

  @jtu.parameterized.parameters(1, 2)
  def test_barrier_arrive_with_predicate(self, arriving_thread):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((), jnp.int32),
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        num_threads=3,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, smem_ref, barrier):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        plgpu.barrier_wait(barrier)
        out_ref[...] = smem_ref[...]

      @pl.when(thread_id > 0)
      def _():
        @pl.when(thread_id == 1)
        def _():
          smem_ref[...] = jnp.int32(42)

        plgpu.barrier_arrive(
            barrier, predicate=thread_id == arriving_thread
        )

    if arriving_thread == 1:
      self.assertEqual(_kernel(), 42)
    else:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
        _kernel().block_until_ready()

  # Test adapted from
  # https://docs.jax.dev/en/latest/pallas/gpu/reference.html#explicit-arrival-cross-thread-synchronization
  #
  # `buffer_size` has to be at least 2 so that the sizes of the `produced` and
  # `consumed` barriers are at least 2 (for each barrier). Otherwise the
  # indexing `produced.at[_]`/`consumed.at[_]` will not work.
  @jtu.parameterized.product(
      skip_floating_point_ops=[False, True],
      input_size=[1, 2, 4, 16, 64, 128],
      buffer_size=[2, 4, 8, 16],
  )
  def test_barrier_for_buffering(
      self, skip_floating_point_ops, input_size, buffer_size, seed=0
  ):
    k = jax.random.key(seed)
    x = jax.random.normal(k, (input_size,), dtype=jnp.float32)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            queue=plgpu.SMEM((buffer_size,), jnp.float32),
            produced=plgpu.Barrier(num_arrivals=1, num_barriers=buffer_size),
            consumed=plgpu.Barrier(num_arrivals=1, num_barriers=buffer_size),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(
            detect_races=True,
            skip_floating_point_ops=skip_floating_point_ops,
        ),
    )
    def _kernel(x_ref, out_ref, queue, produced, consumed):
      thread_id = jax.lax.axis_index('t')
      _get_slot = lambda i: jax.lax.rem(i, buffer_size)

      def _thread0_body(i, _):
        slot = _get_slot(i)
        @pl.when(i >= buffer_size)
        def _await_consumed():
          plgpu.barrier_wait(consumed.at[slot])
        queue[slot] = 3.0 * x_ref[i]
        plgpu.barrier_arrive(produced.at[slot])

      pl.when(thread_id == 0)(
          lambda: jax.lax.fori_loop(0, input_size, _thread0_body, None)
      )

      def _thread1_body(i, _):
        slot = _get_slot(i)
        plgpu.barrier_wait(produced.at[slot])
        out_ref[i] = queue[slot] + 42.0
        plgpu.barrier_arrive(consumed.at[slot])

      pl.when(thread_id == 1)(
          lambda: jax.lax.fori_loop(0, input_size, _thread1_body, None)
      )

      # TODO(nrink): This epilogue is needed to satisfy the requirement that all
      # completed barrier arrivals must have been waited for by the end of
      # kernel execution. Relax this requirement to match what is required when
      # the kernel is executed on real GPU hardware, where having unawaited
      # completed arrivals is acceptable immediately before returning from the
      # kernel.
      def _thread0_epilogue(i, _):
        slot = _get_slot(i)
        @pl.when(i < buffer_size)
        def _await_consumed():
          plgpu.barrier_wait(consumed.at[slot])

      pl.when(thread_id == 0)(
          lambda: jax.lax.fori_loop(0, input_size, _thread0_epilogue, None)
      )

    y = _kernel(x)
    if skip_floating_point_ops:
      np.testing.assert_array_equal(y, jnp.full_like(y, jnp.inf))
    else:
      np.testing.assert_array_equal(y, 3.0 * x + 42.0)
    # TODO(nrink): Add a variant of this test case that does not correctly use
    # the `consumed` barrier and therefore has a race. Test that the race is
    # detected.

  def test_indexing_singleton_barrier_ok(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=(plgpu.Barrier(num_barriers=1),),
        num_threads=1,
        thread_name='t',
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')
      plgpu.barrier_arrive(barrier_ref.at[thread_id])
      out_ref[...] = 42

    y = _kernel()
    self.assertEqual(y, 42)

  def test_not_indexing_multiple_barriers_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=(plgpu.Barrier(num_barriers=2),),
        num_threads=1,
        thread_name='t',
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      plgpu.barrier_arrive(barrier_ref)
      out_ref[...] = 42

    with self.assertRaisesRegex(
        ValueError,
        r'Attempting to operate on barrier without indexing, but `num_barriers'
        r' = 2`',
    ):
      _kernel()

  def test_wait_for_barrier_twice(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier_ref=plgpu.Barrier(num_arrivals=1, num_barriers=2)
        ),
        num_threads=3,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        out_ref[...] = 1
        plgpu.barrier_wait(barrier_ref.at[0])
        out_ref[...] = 2
        plgpu.barrier_arrive(barrier_ref.at[1])
        plgpu.barrier_wait(barrier_ref.at[0])
        out_ref[...] = 3

      @pl.when(thread_id == 1)
      def _():
        plgpu.barrier_arrive(barrier_ref.at[0])

      @pl.when(thread_id == 2)
      def _():
        plgpu.barrier_wait(barrier_ref.at[1])
        plgpu.barrier_arrive(barrier_ref.at[0])

    y = _kernel()
    self.assertEqual(y, 3)

  def test_completing_barrier_twice_in_same_thread_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(barrier_ref=plgpu.Barrier(num_arrivals=1)),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      plgpu.barrier_arrive(barrier_ref)
      plgpu.barrier_arrive(barrier_ref)
      out_ref[...] = 42

    with self.assertRaisesRegex(
        Exception,
        r'Barrier completed phase 1, but no threads observed phase 0.'
    ):
      _kernel()

  def test_completing_barrier_twice_in_different_threads_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2,), jnp.int32),
        scratch_types=dict(barrier_ref=plgpu.Barrier(num_arrivals=1)),
        interpret=InterpretParams(),
        num_threads=2,
        thread_name='t',
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')
      plgpu.barrier_arrive(barrier_ref)
      out_ref[thread_id] = thread_id

    with self.assertRaisesRegex(
        Exception,
        r'Barrier completed phase 1, but no threads observed phase 0.',
    ):
      _kernel()

  @jtu.parameterized.product(
      num_arriving_threads=list(range(1, 8)),
      num_observing_threads=list(range(1, 8)),
      num_threads=[16],
  )
  def test_barrier_wait_in_multiple_threads_ok(
      self, num_arriving_threads, num_observing_threads, num_threads
  ):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((num_threads,), jnp.int32),
        scratch_types=dict(
            barrier_ref=plgpu.Barrier(
                num_arrivals=num_arriving_threads, num_barriers=1
            )
        ),
        num_threads=num_threads,
        thread_name='t',
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id < num_arriving_threads)
      def _():
        out_ref[thread_id] = thread_id
        plgpu.barrier_arrive(barrier_ref)

      @pl.when(thread_id >= num_threads - num_observing_threads)
      def _():
        out_ref[thread_id] = thread_id
        plgpu.barrier_wait(barrier_ref)

    _kernel()

  @jtu.parameterized.parameters(False, True)
  def test_not_awaiting_final_barrier_phase(self, allocate_again):
    # A thread may skip the final completion of a barrier, unless it then
    # allocates another barrier.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref):
      @functools.partial(
          pl.run_scoped, barrier_ref=plgpu.Barrier(num_arrivals=1)
      )
      def _(barrier_ref):
        plgpu.barrier_arrive(barrier_ref)
        plgpu.barrier_wait(barrier_ref)
        plgpu.barrier_arrive(barrier_ref)

      if allocate_again:
        @functools.partial(
            pl.run_scoped, barrier_ref=plgpu.Barrier(num_arrivals=1)
        )
        def _(barrier_ref):
          del barrier_ref  # Unused.

      out_ref[...] = 42

    if allocate_again:
      with self.assertRaisesRegex(
          Exception,
          r'allocated a barrier after deallocating barrier \d+, but thread .+'
          r' had only observed that barrier up to phase 0, while it completed'
          r' up to phase 1',
      ):
        _kernel()
    else:
      self.assertEqual(_kernel(), 42)

  def test_some_threads_not_awaiting_final_barrier_phase_raises(self):
    num_threads = 8

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        num_threads=num_threads,
        thread_name='t',
        interpret=InterpretParams(),
    )
    def _kernel(out_ref):
      thread_id = jax.lax.axis_index('t')

      @functools.partial(
          pl.run_scoped,
          barrier_ref=plgpu.Barrier(num_arrivals=1),
          observed_ref=plgpu.Barrier(num_arrivals=num_threads - 1),
          collective_axes='t',
      )
      def _(barrier_ref, observed_ref):
        @pl.when(thread_id == 0)
        def _():
          plgpu.barrier_arrive(barrier_ref)
          plgpu.barrier_wait(observed_ref)
          plgpu.barrier_arrive(barrier_ref)

        @pl.when(thread_id != 0)
        def _():
          plgpu.barrier_wait(barrier_ref)
          plgpu.barrier_arrive(observed_ref)

          # Only the last thread skips the second completion.
          @pl.when(thread_id != num_threads - 1)
          def _():
            plgpu.barrier_wait(barrier_ref)

      @functools.partial(
          pl.run_scoped,
          barrier_ref=plgpu.Barrier(num_arrivals=1),
          collective_axes='t',
      )
      def _(barrier_ref):
        del barrier_ref  # Unused.

      out_ref[...] = 42

    with self.assertRaisesRegex(
        Exception,
        rf'Warpgroup\(.+warpgroup_id={num_threads - 1}\) allocated a barrier',
    ):
      _kernel()

  def test_warp_not_awaiting_final_barrier_phase_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref):
      @functools.partial(
          pl.run_scoped, barrier_ref=plgpu.Barrier(num_arrivals=1)
      )
      def _(barrier_ref):
        plgpu.barrier_arrive(barrier_ref)

        @plgpu.warp_map
        def _per_warp(warp_id):
          @pl.when(warp_id == 0)
          def _():
            plgpu.barrier_wait(barrier_ref)

        plgpu.barrier_arrive(barrier_ref)

      @functools.partial(
          pl.run_scoped, barrier_ref=plgpu.Barrier(num_arrivals=1)
      )
      def _(barrier_ref):
        del barrier_ref  # Unused.

      out_ref[...] = 42

    with self.assertRaisesRegex(Exception, r'thread Warp\(.+warp_id=0\)'):
      _kernel()

  @jtu.parameterized.parameters(False, True)
  def test_barrier_completing_after_reallocating_thread_exits(
      self, allocate_again
  ):
    # Thread 0 observes the first completion of `barrier_ref`, deallocates it,
    # optionally allocates another barrier, and then exits. Thread 1 completes
    # `barrier_ref` a second time only after thread 0's final action (arriving
    # at `done_ref`), so thread 0 passes the check when it allocates the other
    # barrier, and its missed completion is only caught when thread 1 finally
    # deallocates `barrier_ref`.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2,), jnp.int32),
        scratch_types=dict(done_ref=plgpu.Barrier(num_arrivals=1)),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, done_ref):
      thread_id = jax.lax.axis_index('t')
      out_ref[thread_id] = 42

      @functools.partial(
          pl.run_scoped,
          barrier_ref=plgpu.Barrier(num_arrivals=1),
          collective_axes='t',
      )
      def _(barrier_ref):
        @pl.when(thread_id == 0)
        def _():
          plgpu.barrier_wait(barrier_ref)

        @pl.when(thread_id == 1)
        def _():
          plgpu.barrier_arrive(barrier_ref)
          plgpu.barrier_wait(done_ref)
          plgpu.barrier_arrive(barrier_ref)

      if allocate_again:
        @functools.partial(
            pl.run_scoped,
            barrier_ref=plgpu.Barrier(num_arrivals=1),
            collective_axes='t',
        )
        def _(barrier_ref):
          del barrier_ref  # Unused.

      @pl.when(thread_id == 0)
      def _():
        plgpu.barrier_arrive(done_ref)

    if allocate_again:
      with self.assertRaisesRegex(
          Exception,
          r'Warpgroup\(.+warpgroup_id=0\) allocated a barrier after'
          r' deallocating barrier \d+, but thread .+ had only observed that'
          r' barrier up to phase 0, while it completed up to phase 1',
      ):
        _kernel()
    else:
      np.testing.assert_array_equal(_kernel(), [42, 42])

  @jtu.parameterized.parameters(False, True)
  def test_barrier_completing_between_deallocation_and_reallocation(
      self, allocate_again
  ):
    # Thread 0 observes the first completion of `barrier_ref` and deallocates
    # it. Thread 1 then completes `barrier_ref` a second time and deallocates
    # it too, and only then does thread 0 (optionally) allocate another barrier
    # and exit. Its missed completion is caught when it allocates the barrier.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2,), jnp.int32),
        scratch_types=dict(
            deallocated_ref=plgpu.Barrier(num_arrivals=1),
            completed_ref=plgpu.Barrier(num_arrivals=1),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, deallocated_ref, completed_ref):
      thread_id = jax.lax.axis_index('t')
      out_ref[thread_id] = 42

      @functools.partial(
          pl.run_scoped,
          barrier_ref=plgpu.Barrier(num_arrivals=1),
          collective_axes='t',
      )
      def _(barrier_ref):
        @pl.when(thread_id == 0)
        def _():
          plgpu.barrier_wait(barrier_ref)

        @pl.when(thread_id == 1)
        def _():
          plgpu.barrier_arrive(barrier_ref)
          plgpu.barrier_wait(deallocated_ref)
          plgpu.barrier_arrive(barrier_ref)

      @pl.when(thread_id == 0)
      def _():
        plgpu.barrier_arrive(deallocated_ref)
        plgpu.barrier_wait(completed_ref)

      @pl.when(thread_id == 1)
      def _():
        plgpu.barrier_arrive(completed_ref)

      if allocate_again:
        @functools.partial(
            pl.run_scoped,
            barrier_ref=plgpu.Barrier(num_arrivals=1),
            collective_axes='t',
        )
        def _(barrier_ref):
          del barrier_ref  # Unused.

    if allocate_again:
      with self.assertRaisesRegex(
          Exception,
          r'Warpgroup\(.+warpgroup_id=0\) allocated a barrier after'
          r' deallocating barrier \d+, but thread .+ had only observed that'
          r' barrier up to phase 0, while it completed up to phase 1',
      ):
        _kernel()
    else:
      np.testing.assert_array_equal(_kernel(), [42, 42])

  @jtu.parameterized.named_parameters(
      ('full_slice_of_one', 1, lambda b, i: b.at[()]),
      ('static_unit_slice', 2, lambda b, i: b.at[pl.ds(1, 1)]),
      ('dynamic_unit_slice', 2, lambda b, i: b.at[pl.ds(i, 1)]),
  )
  def test_barrier_slice_names_one_barrier(self, num_barriers, select):
    # As when lowering, a slice of a barrier array contributes its start:
    # `barrier.at[()]` on a single barrier and `barrier.at[pl.ds(i, 1)]` both
    # mean `barrier.at[i]`.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 64), jnp.float32),
        scratch_types=dict(
            smem=plgpu.SMEM((128, 64), jnp.float32),
            barrier=plgpu.Barrier(num_barriers=num_barriers),
        ),
        interpret=InterpretParams(),
    )
    def kernel(i_ref, x_ref, o_ref, smem, barrier):
      b = select(barrier, i_ref[0])
      plgpu.copy_gmem_to_smem(x_ref, smem, b)
      plgpu.barrier_wait(b)
      o_ref[...] = smem[...] + 1.0

    x = jnp.arange(128 * 64, dtype=jnp.float32).reshape(128, 64)
    y = kernel(jnp.array([num_barriers - 1], jnp.int32), x)
    np.testing.assert_array_equal(y, np.asarray(x) + 1.0)

  def test_not_waiting_for_all_barrier_completions_in_thread_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier_ref=plgpu.Barrier(num_arrivals=1, num_barriers=2)
        ),
        num_threads=3,
        thread_name='t',
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        plgpu.barrier_arrive(barrier_ref.at[0])

      @pl.when(thread_id == 1)
      def _():
        # This thread observes only the first completed arrival at
        # `barrier_ref.at[0]`
        plgpu.barrier_wait(barrier_ref.at[0])
        plgpu.barrier_arrive(barrier_ref.at[0])
        plgpu.barrier_arrive(barrier_ref.at[1])

      @pl.when(thread_id == 2)
      def _():
        plgpu.barrier_wait(barrier_ref.at[1])
        # This thread observes only the second completed arrival at
        # `barrier_ref.at[0]`
        plgpu.barrier_wait(barrier_ref.at[0])
        out_ref[...] = 42

    with self.assertRaisesRegex(
        Exception,
        r'Thread Warpgroup\(.+\) is waiting at barrier \w+ for the first time, but barrier is already at phase 2'
    ):
      _kernel()

  @jtu.parameterized.product(
      num_blocks_w=[1, 2, 3],
      num_blocks_x=[1, 2, 3],
      num_blocks_y=[1, 2, 3],
      num_threads=[1, 2, 3],
  )
  def test_grid_iteration(
      self, num_blocks_w, num_blocks_x, num_blocks_y, num_threads
  ):
    def _kernel(a_gmem, out_gmem):
      w = jax.lax.axis_index('w')
      x = jax.lax.axis_index('x')
      y = jax.lax.axis_index('y')
      z = jax.lax.axis_index('z')

      offset = (
          w * num_blocks_x * num_blocks_y * num_threads
          + x * num_blocks_y * num_threads
          + y * num_threads
          + z
      )
      out_gmem[w, x, y, z] = a_gmem[w, x, y, z] + offset

    a = 42 * jnp.ones(
        (num_blocks_w, num_blocks_x, num_blocks_y, num_threads),
        dtype=jnp.int32,
    )

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(a),
        grid=(num_blocks_w, num_blocks_x, num_blocks_y),
        grid_names=('w', 'x', 'y'),
        num_threads=num_threads,
        thread_name='z',
        interpret=InterpretParams(detect_races=True),
    )

    y = kernel(a)
    expected = a + jnp.arange(
        num_blocks_w * num_blocks_x * num_blocks_y * num_threads,
        dtype=a.dtype,
    ).reshape(a.shape)
    np.testing.assert_array_equal(y, expected)

  @jtu.parameterized.product(
      tile_x=[1, 2, 4],
      tile_y=[1, 2, 4],
      tile_z=[1, 2, 4],
      swap_grid_axes=[True, False],
  )
  def test_add_over_grid(self, tile_x, tile_y, tile_z, swap_grid_axes):
    dtype = jnp.int32
    x, y, z = 4, 4, 4

    assert x % tile_x == 0
    assert y % tile_y == 0
    assert z % tile_z == 0

    a = jnp.arange(x * y * z, dtype=dtype).reshape((x, y, z))
    b = jnp.ones((x, y, z), dtype=dtype)

    x_iters = x // tile_x
    y_iters = y // tile_y
    z_iters = z // tile_z

    def _kernel(a_gmem, b_gmem, out_gmem):
      xi = jax.lax.axis_index('x')
      yi = jax.lax.axis_index('y')
      zi = jax.lax.axis_index('z')
      xi_slice = pl.ds(xi * tile_x, tile_x)
      yi_slice = pl.ds(yi * tile_y, tile_y)
      zi_slice = pl.ds(zi * tile_z, tile_z)

      out_gmem[xi_slice, yi_slice, zi_slice] = (
          a_gmem[xi_slice, yi_slice, zi_slice]
          + b_gmem[xi_slice, yi_slice, zi_slice]
      )

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct((x, y, z), dtype),
        grid=_maybe_reverse((x_iters, y_iters), swap_grid_axes),
        grid_names=_maybe_reverse(('x', 'y'), swap_grid_axes),
        num_threads=z_iters,
        thread_name='z',
        interpret=InterpretParams(detect_races=True),
    )

    expected = a + b
    y = kernel(a, b)
    np.testing.assert_array_equal(y, expected)

  @jtu.parameterized.product(
      tile_m=[1, 2, 4],
      tile_k=[1, 2, 4],
      tile_n=[1, 2, 4],
      swap_grid_axes=[True, False],
  )
  def test_matmul_over_grid_with_barrier_and_smem(
      self, tile_m, tile_k, tile_n, swap_grid_axes
  ):
    dtype = jnp.int32
    m, k, n = 4, 4, 4

    assert m % tile_m == 0
    assert k % tile_k == 0
    assert n % tile_n == 0

    a = jnp.arange(16, dtype=dtype).reshape((m, k))
    b = jnp.ones((k, n), dtype=dtype)

    m_iters = m // tile_m
    k_iters = k // tile_k
    n_iters = n // tile_n

    def _kernel(a_gmem, b_gmem, out_gmem, acc_smem, barrier):
      mi = jax.lax.axis_index('m')
      ki = jax.lax.axis_index('k')
      ni = jax.lax.axis_index('n')
      mi_slice = pl.ds(mi * tile_m, tile_m)
      ki_slice = pl.ds(ki * tile_k, tile_k)
      ni_slice = pl.ds(ni * tile_n, tile_n)

      # We map the reduced dimension, i.e. `k`, to the thread dimension. This
      # allows us to do the accumulation into an `SMEM` buffer, i.e. `acc_smem`.
      # (Note that a fresh `SMEM` buffer is allocated for each grid point, but
      # for each fixed grid point, a single `SMEM` buffer is shared across the
      # threads.) We then need to use barriers to sequentialize access to the
      # `SMEM` buffer between the threads. (Note that the specific sequential
      # order of the updates to `acc_smem` does not matter, so long as there are
      # no races.)

      @pl.when(ki == 0)
      def _():
        acc_smem[...] = jnp.zeros(acc_smem.shape, dtype=acc_smem.dtype)
        plgpu.barrier_arrive(barrier.at[0])

      plgpu.barrier_wait(barrier.at[ki])
      # TODO(nrink): Matrix multiplication with `@` is not supported for real
      # GPU kernels (but the GPU kernel interpreter allows this). Replace this
      # with a `wgmma` or `tcgen05_mma` once these are supported by the GPU
      # kernel interpreter.
      acc_smem[...] += a_gmem[mi_slice, ki_slice] @ b_gmem[ki_slice, ni_slice]
      plgpu.barrier_arrive(barrier.at[(ki + 1) % k_iters])

      @pl.when(ki == 0)
      def _():
        plgpu.barrier_wait(barrier.at[0])
        out_gmem[mi_slice, ni_slice] = acc_smem[...]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct((m, n), dtype),
        scratch_types=dict(
            acc_smem=plgpu.SMEM((tile_m, tile_n), dtype),
            barrier=plgpu.Barrier(num_barriers=k_iters),
        ),
        grid=_maybe_reverse((m_iters, n_iters), swap_grid_axes),
        grid_names=_maybe_reverse(('m', 'n'), swap_grid_axes),
        num_threads=k_iters,
        thread_name='k',
        interpret=InterpretParams(detect_races=True),
    )

    expected = a @ b
    y = kernel(a, b)
    np.testing.assert_array_equal(y, expected)

  def test_matmul_over_grid_with_race(self, tile_m=4, tile_k=2, tile_n=4):
    dtype = jnp.int32
    m, k, n = 4, 4, 4

    assert m % tile_m == 0
    assert k % tile_k == 0
    assert n % tile_n == 0

    a = jnp.arange(16, dtype=dtype).reshape((m, k))
    b = jnp.ones((k, n), dtype=dtype)

    m_iters = m // tile_m
    k_iters = k // tile_k
    n_iters = n // tile_n

    def _kernel(a_gmem, b_gmem, _, acc_smem):
      mi = jax.lax.axis_index('m')
      ki = jax.lax.axis_index('k')
      ni = jax.lax.axis_index('n')
      mi_slice = pl.ds(mi * tile_m, tile_m)
      ki_slice = pl.ds(ki * tile_k, tile_k)
      ni_slice = pl.ds(ni * tile_n, tile_n)

      # The two threads race to update `acc_smem`. We do not bother with
      # initializing `acc_mem` to zero, nor with copying the final result out to
      # `GMEM`, as is done in the correct (i.e. race-free) test above (i.e. in
      # `test_matmul_over_grid_with_barrier_and_smem`). This would only
      # introduce additional races.
      #
      # TODO(nrink): Matrix multiplication with `@` is not supported for real
      # GPU kernels (but the GPU kernel interpreter allows this). Replace this
      # with a `wgmma` or `tcgen05_mma` once these are supported by the GPU
      # kernel interpreter.
      acc_smem[...] += a_gmem[mi_slice, ki_slice] @ b_gmem[ki_slice, ni_slice]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct((m, n), dtype),
        scratch_types=dict(
            acc_smem=plgpu.SMEM((tile_m, tile_n), dtype),
        ),
        grid=(m_iters, n_iters),
        grid_names=('m', 'n'),
        num_threads=k_iters,
        thread_name='k',
        interpret=InterpretParams(detect_races=True),
    )

    with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
      kernel(a, b).block_until_ready()

  @jtu.parameterized.product(with_race=[True, False])
  def test_copy_gmem_to_smem_single_thread(self, with_race):
    x = jnp.arange(16, dtype=jnp.int32).reshape((2, 8))

    def _kernel(in_gmem, out_gmem, barrier, smem):
      plgpu.copy_gmem_to_smem(in_gmem, smem, barrier)
      if not with_race:
        plgpu.barrier_wait(barrier)
      out_gmem[...] = smem[...]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            barrier=plgpu.Barrier(), smem=plgpu.SMEM(x.shape, x.dtype)
        ),
    )

    if with_race:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
        kernel(x).block_until_ready()
    else:
      y = kernel(x)
      np.testing.assert_array_equal(y, x)

  @jtu.parameterized.product(with_race=[True, False])
  def test_copy_gmem_to_smem_two_threads(self, with_race):
    x = jnp.arange(16, dtype=jnp.int32).reshape((2, 8))

    def _kernel(in_gmem, out_gmem, barrier, smem):
      tid = jax.lax.axis_index('t')

      @pl.when(tid == 0)
      def _():
        plgpu.copy_gmem_to_smem(in_gmem, smem, barrier)

      @pl.when(tid == 1)
      def _():
        if not with_race:
          plgpu.barrier_wait(barrier)
        out_gmem[...] = smem[...]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            barrier=plgpu.Barrier(),
            smem=plgpu.SMEM(x.shape, x.dtype),
        ),
        num_threads=2,
        thread_name='t',
    )

    if with_race:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
        kernel(x).block_until_ready()
    else:
      y = kernel(x)
      np.testing.assert_array_equal(y, x)

  @jtu.parameterized.product(with_race=[True, False])
  def test_copy_gmem_to_smem_parallel_two_threads(self, with_race):
    x = jnp.arange(16, dtype=jnp.int32).reshape((2, 8))

    def _kernel(in_gmem, out_gmem, per_thread_barrier, smem0, smem1):
      tid = jax.lax.axis_index('t')

      def _per_thread_kernel(smem):
        plgpu.copy_gmem_to_smem(in_gmem, smem, per_thread_barrier.at[tid])
        if not with_race:
          plgpu.barrier_wait(per_thread_barrier.at[tid])
        out_gmem[tid, ...] = smem[tid, ...]

      # TODO(nrink): Investigate why this does not work with a single SMEM
      # buffer and `smem.at[0]` and `smem.at[1]`.
      pl.when(tid == 0)(functools.partial(_per_thread_kernel, smem=smem0))
      pl.when(tid == 1)(functools.partial(_per_thread_kernel, smem=smem1))

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            per_thread_barrier=plgpu.Barrier(num_barriers=2),
            smem0=plgpu.SMEM(x.shape, x.dtype),
            smem1=plgpu.SMEM(x.shape, x.dtype),
        ),
        num_threads=2,
        thread_name='t',
    )

    if with_race:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
        kernel(x).block_until_ready()
    else:
      y = kernel(x)
      np.testing.assert_array_equal(y, x)

  @jtu.parameterized.product(num_tma_threads_per_device=[2, 3, 4])
  def test_copy_gmem_to_smem_multiple_tma_threads(
      self, num_tma_threads_per_device
  ):
    x = jnp.arange(16, dtype=jnp.int32).reshape((2, 8))
    y = 2 * x

    def _kernel(gmem0, gmem1, out_gmem, barrier, smem):
      plgpu.copy_gmem_to_smem(gmem0, smem, barrier.at[0])
      plgpu.copy_gmem_to_smem(gmem1, smem, barrier.at[1])
      plgpu.barrier_wait(barrier.at[0])
      plgpu.barrier_wait(barrier.at[1])
      out_gmem[...] = gmem0[...] + gmem1[...]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(
            detect_races=True,
            num_tma_threads_per_device=num_tma_threads_per_device,
        ),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_barriers=2),
            smem=plgpu.SMEM(x.shape, x.dtype),
        ),
    )

    with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
      kernel(x, y).block_until_ready()

  @jtu.parameterized.product(with_race=[True, False])
  def test_copy_gmem_to_smem_multiple_arrivals_at_barrier(self, with_race):
    x = jnp.arange(16, dtype=jnp.int32).reshape((2, 8))
    y = 2 * x

    def _kernel(gmem0, gmem1, out_gmem, barrier, smem0, smem1):
      plgpu.copy_gmem_to_smem(gmem0, smem0, barrier)
      plgpu.copy_gmem_to_smem(gmem1, smem1, barrier)
      if not with_race:
        plgpu.barrier_wait(barrier)
      out_gmem[...] = smem0[...] + smem1[...]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=2),
            smem0=plgpu.SMEM(x.shape, x.dtype),
            smem1=plgpu.SMEM(x.shape, x.dtype),
        ),
    )

    if with_race:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
        kernel(x, y).block_until_ready()
    else:
      z = kernel(x, y)
      np.testing.assert_array_equal(z, x + y)

  def test_copy_smem_to_gmem(self):
    def _kernel(out_gmem, smem_ref):
      tid = jax.lax.axis_index('t')
      i = jax.lax.axis_index('i')
      j = jax.lax.axis_index('j')

      smem_ref[tid, pl.ds(i*16, 16), pl.ds(j*32, 32)] = (
          jnp.arange((2*32*128), dtype=jnp.int32).reshape((2, 32, 128))[
              tid, pl.ds(i*16, 16), pl.ds(j*32, 32)])
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          smem_ref.at[tid, pl.ds(i*16, 16), pl.ds(j*32, 32)],
          out_gmem.at[tid, pl.ds(i*16, 16), pl.ds(j*32, 32)])
      plgpu.wait_smem_to_gmem(0)

    y = jax.jit(plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct((2, 32, 128), jnp.int32),
        interpret=InterpretParams(),
        grid=(2, 4),
        grid_names=('i', 'j'),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((2, 32, 128), jnp.int32)
        ),
        num_threads=2,
        thread_name='t',
    ))()

    expected = jnp.arange((2*32*128), dtype=jnp.int32).reshape((2, 32, 128))
    np.testing.assert_array_equal(y, expected)

  @jtu.parameterized.product(
      grid_dict=[
          None,
          dict(g0=1),
          dict(g0=2, g1=3),
      ],
      cluster_dict=[
          None,
          dict(c0=2),
          dict(c0=2, c1=3),
          dict(c0=2, c1=3, c2=2),
      ],
      thread_dict=[
          None,
          dict(t=1),
          dict(t=3),
      ],
  )
  def test_cluster(self, grid_dict, cluster_dict, thread_dict):
    mesh_kwargs = {}
    axes_dims = ()
    axes_names = ()
    if grid_dict is not None:
      mesh_kwargs['grid'] = tuple(grid_dict.values())
      mesh_kwargs['grid_names'] = tuple(grid_dict.keys())
      axes_dims += tuple(grid_dict.values())
      axes_names += tuple(grid_dict.keys())
    if cluster_dict is not None:
      mesh_kwargs['cluster'] = tuple(cluster_dict.values())
      mesh_kwargs['cluster_names'] = tuple(cluster_dict.keys())
      axes_dims += tuple(cluster_dict.values())
      axes_names += tuple(cluster_dict.keys())
    if thread_dict is not None:
      (thread_name, num_threads), = thread_dict.items()
      mesh_kwargs['num_threads'] = num_threads
      mesh_kwargs['thread_name'] = thread_name
      axes_dims += (num_threads,)
      axes_names += (thread_name,)
    out_shape = axes_dims

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(out_shape, jnp.int32),
        interpret=InterpretParams(detect_races=True),
        **mesh_kwargs,
    )
    def kernel(o_ref):
      flat_thread_id = jnp.int32(0)
      for i, name in enumerate(axes_names):
        stride = math.prod(axes_dims[i + 1 :])
        flat_thread_id += jax.lax.axis_index(name) * stride
      thread_idx = tuple(jax.lax.axis_index(name) for name in axes_names)

      o_ref[thread_idx] = flat_thread_id

    expected = np.arange(
        math.prod(out_shape), dtype=jnp.int32
    ).reshape(out_shape)
    y = kernel()
    np.testing.assert_array_equal(y, expected)

  def test_different_blocks_dont_share_memory(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2,), jnp.int32),
        scratch_types=dict(
          smem_ref=plgpu.SMEM((), jnp.int32),
        ),
        cluster=(2,),
        cluster_names=('c',),
        interpret=InterpretParams()
    )
    def _kernel(in_ref, out_ref, smem_ref):
      smem_ref[...] = in_ref[jax.lax.axis_index('c')]
      out_ref[jax.lax.axis_index('c')] = smem_ref[...]

    x = jnp.arange(2, dtype=jnp.int32)
    output = _kernel(x)
    np.testing.assert_array_equal(output, x)

  def test_warp_specialize(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((5,), jnp.int32),
        interpret=InterpretParams(),
    )
    def _kernel(in_ref, out_ref):
      out_ref[...] = jnp.zeros((5,), jnp.int32)

      @plgpu.warp_map
      def _per_warp(warp_id):

        @pl.when(warp_id == 0)
        def _():
          out_ref[0] = in_ref[0]

        @pl.when(warp_id == 1)
        def _():
          out_ref[1] = in_ref[1]

        @pl.when(warp_id == 2)
        def _():
          out_ref[2] = in_ref[2]

        @pl.when(warp_id == 3)
        def _():
          out_ref[3] = in_ref[3]

        # this should never be reached
        @pl.when(warp_id == 4)
        def _():
          out_ref[4] = in_ref[4]

    x = jnp.arange(5, dtype=jnp.int32)
    expected = x.at[4].set(0)
    output = _kernel(x)
    np.testing.assert_array_equal(output, expected)

  def test_warps_share_shared_memory(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((), jnp.int32),
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, smem_ref, barrier):

      @plgpu.warp_map
      def _per_warp(warp_id):

        @pl.when(warp_id == 0)
        def _():
          smem_ref[...] = 1
          plgpu.barrier_arrive(barrier)

        @pl.when(warp_id == 1)
        def _():
          plgpu.barrier_wait(barrier)
          out_ref[...] = smem_ref[...]

    out = _kernel()
    self.assertEqual(out, 1)

  def test_cant_warp_specialize_twice(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref):
      @plgpu.warp_map
      def _per_warp(_warp_id):
        @plgpu.warp_map
        def _(_warp_id):
          out_ref[...] = 42

    with self.assertRaisesRegex(
        Exception, r'Cannot mpmd_map over WarpMesh while already mpmd_mapped'
    ):
      _kernel()

  def test_all_warps_must_wait_on_barrier(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(_out_ref, barrier):
      plgpu.barrier_arrive(barrier)

      # only three warps wait on barrier
      @plgpu.warp_map
      def _per_warp(warp_id):
        @pl.when(warp_id != 0)
        def _():
          plgpu.barrier_wait(barrier)

      # now the whole warpgroup waits on the barrier
      plgpu.barrier_wait(barrier)

    with self.assertRaisesRegex(
        Exception,
        r'Warpgroup-thread Warpgroup\(.+\) is waiting at barrier \d+, but only'
        r' 3 of its constituent warps have participated in the barrier',
    ):
      _kernel()

  def test_warps_must_wait_on_barrier_evenly(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(_out_ref, barrier):
      plgpu.barrier_arrive(barrier)
      @plgpu.warp_map
      def _per_warp(warp_id):
        @pl.when(warp_id != 3)
        def _():
          plgpu.barrier_wait(barrier)

      plgpu.barrier_arrive(barrier)
      @plgpu.warp_map
      def _per_warp(warp_id):
        @pl.when(warp_id != 0)
        def _():
          plgpu.barrier_wait(barrier)

    with self.assertRaisesRegex(
        Exception,
        r'Thread Warp\(.+\) is waiting at barrier \d+ for the first time, but'
        r' barrier is already at phase 2.'
    ):
      _kernel()

  def test_warps_of_warpgroup_must_wait_on_barrier_evenly(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(_out_ref, barrier):
      plgpu.barrier_arrive(barrier)

      # all four warps wait on the barrier
      @plgpu.warp_map
      def _per_warp(warp_id):
        del warp_id
        plgpu.barrier_wait(barrier)

      plgpu.barrier_arrive(barrier)

      # only three warps wait on barrier
      @plgpu.warp_map
      def _per_warp(warp_id):
        @pl.when(warp_id != 0)
        def _():
          plgpu.barrier_wait(barrier)

      # now the whole warpgroup waits on the barrier
      plgpu.barrier_wait(barrier)

    with self.assertRaisesRegex(
        Exception,
        r'Warpgroup-thread Warpgroup\(.+\) is waiting at barrier \d+, but its'
        r' constituent warps have previously observed different phases',
    ):
      _kernel()

  def test_warps_get_parent_warpgroups_barrier_phase(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier):
      plgpu.barrier_arrive(barrier)
      plgpu.barrier_wait(barrier)
      plgpu.barrier_arrive(barrier)
      plgpu.barrier_wait(barrier)
      plgpu.barrier_arrive(barrier)

      @plgpu.warp_map
      def _per_warp(warp_id):
        del warp_id
        plgpu.barrier_wait(barrier)

      out_ref[...] = 42

    out = _kernel()
    self.assertEqual(out, 42)

  def test_warpgroup_gets_warps_barrier_phase(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier):
      plgpu.barrier_arrive(barrier)

      @plgpu.warp_map
      def _per_warp(warp_id):
        del warp_id
        plgpu.barrier_wait(barrier)

      plgpu.barrier_arrive(barrier)

      @plgpu.warp_map
      def _per_warp(warp_id):
        del warp_id
        plgpu.barrier_wait(barrier)

      # warpgroup can wait, since its consituent warps observed prior phases
      plgpu.barrier_arrive(barrier)
      plgpu.barrier_wait(barrier)
      out_ref[...] = 42

    out = _kernel()
    self.assertEqual(out, 42)

  def test_mpmd_warp_specialize(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((5,), jnp.int32),
        interpret=InterpretParams(),
    )
    def _kernel(in_ref, out_ref):
      out_ref[...] = jnp.zeros((5,), jnp.int32)

      def _per_warp():
        warp_id = jax.lax.axis_index('warp')

        @pl.when(warp_id == 0)
        def _():
          out_ref[0] = in_ref[0]

        @pl.when(warp_id == 1)
        def _():
          out_ref[1] = in_ref[1]

        @pl.when(warp_id == 2)
        def _():
          out_ref[2] = in_ref[2]

        @pl.when(warp_id == 3)
        def _():
          out_ref[3] = in_ref[3]

        # this should never be reached
        @pl.when(warp_id == 4)
        def _():
          out_ref[4] = in_ref[4]

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp'), _per_warp)])()

    x = jnp.arange(5, dtype=jnp.int32)
    expected = x.at[4].set(0)
    output = _kernel(x)
    np.testing.assert_array_equal(output, expected)

  def test_mpmd_warps_share_shared_memory(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((), jnp.int32),
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, smem_ref, barrier):
      def _per_warp():
        warp_id = jax.lax.axis_index('warp')

        @pl.when(warp_id == 0)
        def _():
          smem_ref[...] = 1
          plgpu.barrier_arrive(barrier)

        @pl.when(warp_id == 1)
        def _():
          plgpu.barrier_wait(barrier)
          out_ref[...] = smem_ref[...]

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp'), _per_warp)])()

    out = _kernel()
    self.assertEqual(out, 1)

  def test_cant_mpmd_map_twice(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref):
      def _per_warp():
        def _():
          out_ref[...] = 42

        mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp1'), _)])()

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp0'), _per_warp)])()

    with self.assertRaisesRegex(
        Exception, r'Cannot mpmd_map over WarpMesh while already mpmd_mapped'
    ):
      _kernel()

  def test_mpmd_all_warps_must_wait_on_barrier(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(_out_ref, barrier):
      plgpu.barrier_arrive(barrier)

      # only three warps wait on barrier
      def _per_warp():
        @pl.when(jax.lax.axis_index('warp') != 0)
        def _():
          plgpu.barrier_wait(barrier)

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp'), _per_warp)])()

      # now the whole warpgroup waits on the barrier
      plgpu.barrier_wait(barrier)

    with self.assertRaisesRegex(
        Exception,
        r'Warpgroup-thread Warpgroup\(.+\) is waiting at barrier \d+, but only'
        r' 3 of its constituent warps have participated in the barrier',
    ):
      _kernel()

  def test_mpmd_warps_must_wait_on_barrier_evenly(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(_out_ref, barrier):
      plgpu.barrier_arrive(barrier)
      def _per_warp():
        @pl.when(jax.lax.axis_index('warp') != 3)
        def _():
          plgpu.barrier_wait(barrier)

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp'), _per_warp)])()

      plgpu.barrier_arrive(barrier)
      def _per_warp():
        @pl.when(jax.lax.axis_index('warp') != 0)
        def _():
          plgpu.barrier_wait(barrier)

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp'), _per_warp)])()

    with self.assertRaisesRegex(
        Exception,
        r'Thread Warp\(.+\) is waiting at barrier \d+ for the first time, but'
        r' barrier is already at phase 2.'
    ):
      _kernel()

  def test_mpmd_warps_of_warpgroup_must_wait_on_barrier_evenly(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(_out_ref, barrier):
      plgpu.barrier_arrive(barrier)

      # all four warps wait on the barrier
      def _per_warp():
        plgpu.barrier_wait(barrier)

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp'), _per_warp)])()

      plgpu.barrier_arrive(barrier)

      # only three warps wait on barrier
      def _per_warp():
        @pl.when(jax.lax.axis_index('warp') != 0)
        def _():
          plgpu.barrier_wait(barrier)

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp'), _per_warp)])()

      # now the whole warpgroup waits on the barrier
      plgpu.barrier_wait(barrier)

    with self.assertRaisesRegex(
        Exception,
        r'Warpgroup-thread Warpgroup\(.+\) is waiting at barrier \d+, but its'
        r' constituent warps have previously observed different phases',
    ):
      _kernel()

  def test_mpmd_warps_get_parent_warpgroups_barrier_phase(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier):
      plgpu.barrier_arrive(barrier)
      plgpu.barrier_wait(barrier)
      plgpu.barrier_arrive(barrier)
      plgpu.barrier_wait(barrier)
      plgpu.barrier_arrive(barrier)

      def _per_warp():
        plgpu.barrier_wait(barrier)

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp'), _per_warp)])()

      out_ref[...] = 42

    out = _kernel()
    self.assertEqual(out, 42)

  def test_mpmd_warpgroup_gets_warps_barrier_phase(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier):
      plgpu.barrier_arrive(barrier)

      def _per_warp():
        plgpu.barrier_wait(barrier)

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp'), _per_warp)])()

      plgpu.barrier_arrive(barrier)

      def _per_warp():
        plgpu.barrier_wait(barrier)

      mpmd.mpmd_map([(plgpu.WarpMesh(axis_name='warp'), _per_warp)])()

      # warpgroup can wait, since its consituent warps observed prior phases
      plgpu.barrier_arrive(barrier)
      plgpu.barrier_wait(barrier)
      out_ref[...] = 42

    out = _kernel()
    self.assertEqual(out, 42)

  def test_async_copy_tmem_with_mma(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,128), jnp.float32),
        scratch_types=dict(
            a_smem=plgpu.SMEM((128,128), jnp.float16),
            a_tmem=plgpu.TMEM((128,128), jnp.float16, packed=True),
            b_smem=plgpu.SMEM((128,128), jnp.float16),
            acc_ref=plgpu.TMEM((128,128), jnp.float32),
            barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(),
    )
    def kernel(a, b, out_ref, a_smem, a_tmem, b_smem, acc_ref, barrier):
      a_smem[...] = a[...]
      b_smem[...] = b[...]
      plgpu.commit_smem()
      plgpu.async_copy_smem_to_tmem(a_smem, a_tmem)
      plgpu.tcgen05_mma(acc_ref, a_tmem, b_smem, accumulate=False)
      plgpu.tcgen05_commit_arrive(barrier)
      plgpu.barrier_wait(barrier)
      out_ref[...] = plgpu.async_load_tmem(acc_ref)

    a = jax.random.uniform(jax.random.key(0), (128, 128), jnp.float16)
    b = jax.random.uniform(jax.random.key(1), (128, 128), jnp.float16)
    output = kernel(a, b)
    expected = np.matmul(np.asarray(a), np.asarray(b), dtype=np.float32)
    self.assertArraysEqual(output, expected)

  def test_async_store_load_tmem(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,128), jnp.int32),
        scratch_types=dict(
            tmem_ref=plgpu.TMEM((128,128), jnp.int32),
        ),
        interpret=InterpretParams(),
    )
    def kernel(in_ref, out_ref, tmem_ref):
      plgpu.async_store_tmem(tmem_ref, in_ref[...])
      plgpu.commit_tmem()
      out_ref[...] = plgpu.async_load_tmem(tmem_ref)

    a = jnp.full((128,128), 42, dtype=jnp.int32)
    output = kernel(a)
    self.assertArraysEqual(output, a)

  @jtu.parameterized.product(grid=[(4,), (2, 3)])
  def test_dynamic_scheduling_loop(self, grid):
    # `dynamic_scheduling_loop` lets a block claim a not-yet-launched block's
    # work. The interpreter enumerates the whole grid itself, so there is never
    # unlaunched work to claim: every cancellation attempt fails -- a legal
    # hardware outcome -- and each block runs exactly its own grid index.
    grid_names = tuple(f'g{i}' for i in range(len(grid)))

    def _kernel(o_ref):
      block = tuple(jax.lax.axis_index(name) for name in grid_names)

      @plgpu.dynamic_scheduling_loop(grid_names, thread_axis='wg')
      def _(loop_info: plgpu.NDLoopInfo):
        o_ref[loop_info.index] = jnp.int32(
            sum(b * s for b, s in zip(block, (*grid[1:], 1)))
        )

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct(grid, jnp.int32),
        grid=grid,
        grid_names=grid_names,
        num_threads=1,
        thread_name='wg',
        interpret=InterpretParams(inline_mgpu='ignore')
    )

    np.testing.assert_array_equal(
        kernel(), np.arange(math.prod(grid), dtype=np.int32).reshape(grid)
    )

  def test_exception_in_kernel_is_reported(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((1,), jnp.float32),
        interpret=InterpretParams(out_of_bounds_reads='raise'),
    )
    def kernel(x_ref, o_ref):
      o_ref[0] = x_ref[100]

    with self.assertRaisesRegex(Exception, r'Out-of-bounds read'):
      kernel(jnp.ones((4,), jnp.float32)).block_until_ready()

    # The failed kernel must not leave any state behind that would break
    # interpreting subsequent kernels.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((4,), jnp.int32),
        interpret=InterpretParams(),
    )
    def _ok_kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1

    x = jnp.arange(4, dtype=jnp.int32)
    np.testing.assert_array_equal(_ok_kernel(x), x + 1)

  def test_exception_wakes_up_thread_waiting_on_barrier(self):
    # Thread 0 waits on a barrier that thread 1 never arrives at, because
    # thread 1 fails first. The failure must wake up thread 0 (instead of
    # hanging forever) and be reported.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((4,), jnp.int32),
        scratch_types=dict(barrier=plgpu.Barrier(num_arrivals=1)),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(out_of_bounds_reads='raise'),
    )
    def _kernel(x_ref, o_ref, barrier):
      t = jax.lax.axis_index('t')

      @pl.when(t == 0)
      def _():
        plgpu.barrier_wait(barrier)
        o_ref[0] = x_ref[0]

      @pl.when(t == 1)
      def _():
        o_ref[1] = x_ref[100]
        plgpu.barrier_arrive(barrier)

    with self.assertRaisesRegex(Exception, r'Out-of-bounds read') as cm:
      _kernel(jnp.arange(4, dtype=jnp.int32)).block_until_ready()
    self.assertRegex(str(cm.exception), r'Computation failed on Warpgroup\(')
    self.assertIn('warpgroup_id=1', str(cm.exception))

  def test_exception_wakes_up_thread_waiting_on_cluster_barrier(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((4,), jnp.int32),
        scratch_types=dict(
            barrier=plgpu.ClusterBarrier(collective_axes=('c',), num_arrivals=1)
        ),
        cluster=(2,),
        cluster_names=('c',),
        interpret=InterpretParams(out_of_bounds_reads='raise'),
    )
    def _kernel(x_ref, o_ref, barrier):
      c = jax.lax.axis_index('c')

      @pl.when(c == 0)
      def _():
        plgpu.barrier_wait(barrier)
        o_ref[0] = x_ref[0]

      @pl.when(c == 1)
      def _():
        o_ref[1] = x_ref[100]
        plgpu.barrier_arrive(barrier)

    with self.assertRaisesRegex(Exception, r'Out-of-bounds read'):
      _kernel(jnp.arange(4, dtype=jnp.int32)).block_until_ready()

  @jtu.parameterized.product(
      num_threads=[1, 2]
  )
  def test_exception_in_warp_map_is_reported(self, num_threads):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((4,), jnp.int32),
        num_threads=num_threads,
        thread_name='t',
        interpret=InterpretParams(out_of_bounds_reads='raise'),
    )
    def _kernel(x_ref, o_ref):
      @plgpu.warp_map
      def _per_warp(warp_id):
        @pl.when((warp_id == 1) & (jax.lax.axis_index('t') == 0))
        def _():
          o_ref[0] = x_ref[100]

    with self.assertRaisesRegex(Exception, r'Out-of-bounds read'):
      _kernel(jnp.arange(4, dtype=jnp.int32)).block_until_ready()

  @jtu.parameterized.product(
      returns_value=[True, False], inline_mgpu=['error', 'ignore', 'fizzbuzz']
  )
  def test_inline_mgpu_handling(self, returns_value, inline_mgpu):
    if returns_value:
      inline = plgpu.inline_mgpu(
          return_type=plgpu.ShapeDtypeStruct((), jnp.float32, None)
      )(lambda _: None)
    else:
      inline = plgpu.inline_mgpu()(lambda _: None)

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        interpret=InterpretParams(inline_mgpu=inline_mgpu),
    )
    def kernel(o_ref):
      value = inline()
      o_ref[...] = jnp.full((128,), 1.0, jnp.float32)
      if returns_value:
        o_ref[...] += value

    if inline_mgpu == 'error':
      with self.assertRaisesRegex(
          ValueError,
          r'GPU interpret mode does not support `inline_mgpu`. Pass'
          r' `inline_mgpu=',
      ):
        kernel()
    elif inline_mgpu == 'ignore':
      if returns_value:
        with self.assertRaisesRegex(
            ValueError,
            r'GPU interpret mode does not support `inline_mgpu` with a return'
            r' value',
        ):
          kernel()
      else:
        np.testing.assert_array_equal(
            kernel(), np.full((128,), 1.0, jnp.float32)
        )
    elif inline_mgpu == 'fizzbuzz':
      with self.assertRaisesRegex(
          ValueError,
          r'GPU interpret mode does not support `inline_mgpu` param with value',
      ):
        kernel()


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  tile_m: int
  tile_n: int
  tile_k: int
  max_concurrent_steps: int
  epilogue_tile_n: int = 64
  grid_minor_dim: int = 0
  grid_tile_width: int = 1


def matmul0(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  m_iters = m // tile_m
  n_iters = n // tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(a_gmem, b_gmem, out_gmem, acc_tmem, acc_smem, consumed_barriers):
    mi = jax.lax.axis_index("m")
    ni = jax.lax.axis_index("n")
    m_slice = pl.ds(mi * tile_m, tile_m)
    n_slice = pl.ds(ni * tile_n, tile_n)

    def do_mma(idxs, a_smem, b_smem):
      (ki,) = idxs
      arrive_barrier_slot = ki % 2
      wait_barrier_slot = 1 - arrive_barrier_slot
      plgpu.tcgen05_mma(
          acc_tmem,
          a_smem,
          b_smem,
          barrier=consumed_barriers.at[arrive_barrier_slot],
          accumulate=(ki > 0),
      )
      plgpu.barrier_wait(consumed_barriers.at[wait_barrier_slot])

    # Make sure the wait succeeds in the first iteration.
    plgpu.barrier_arrive(consumed_barriers.at[1])
    block_kwargs = dict(transforms=transforms, delay_release=1)
    plgpu.emit_pipeline(
      do_mma,
      in_specs=[
          plgpu.BlockSpec((tile_m, tile_k), lambda ki: (mi, ki), **block_kwargs),
          plgpu.BlockSpec((tile_k, tile_n), lambda ki: (ki, ni), **block_kwargs),
      ],
      grid=(k_iters,),
      max_concurrent_steps=max_concurrent_steps,
    )(a_gmem, b_gmem)

    final_barrier = 1 - (k_iters % 2)
    plgpu.barrier_wait(consumed_barriers.at[final_barrier])
    acc_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(acc_smem, out_gmem.at[m_slice, n_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  f = plgpu.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=("m", "n"),
      scratch_types=dict(
          acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
          acc_smem=plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1, num_barriers=2, orders_tensor_core=True
          ),
      ),
  )
  return f(a, b)


def matmul1(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  if m % tile_m != 0:
    raise ValueError(f'{m=} must be divisible by {tile_m=}')
  if n % tile_n != 0:
    raise ValueError(f'{n=} must be divisible by {tile_n=}')
  if k % tile_k != 0:
    raise ValueError(f'{k=} must be divisible by {tile_k=}')
  m_iters = m // tile_m
  n_iters = n // tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(
      a_gmem,
      b_gmem,
      out_gmem,
      a_smem,
      b_smem,
      acc_tmem,
      acc_smem,
      load_barriers,
      consumed_barriers,
      mma_done_barrier,
  ):
    m_index = jax.lax.axis_index('m')
    n_index = jax.lax.axis_index('n')
    m_slice = pl.ds(m_index * tile_m, tile_m)
    n_slice = pl.ds(n_index * tile_n, tile_n)

    @plgpu.warp_map
    def _per_warp(warp_id):

      @pl.when(warp_id == 0)
      def _memory():
        def _loop_body(ki, _):
          slot = jax.lax.rem(ki, max_concurrent_steps)

          @pl.when(ki >= max_concurrent_steps)
          def _():  # Make sure the data has been consumed before overwriting.
            plgpu.barrier_wait(consumed_barriers.at[slot])

          k_slice = pl.ds(ki * tile_k, tile_k)
          plgpu.copy_gmem_to_smem(
              a_gmem.at[m_slice, k_slice],
              a_smem.at[slot],
              load_barriers.at[slot],
          )
          plgpu.copy_gmem_to_smem(
              b_gmem.at[k_slice, n_slice],
              b_smem.at[slot],
              load_barriers.at[slot],
          )

        jax.lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(warp_id == 1)
      def _compute():
        def _loop_body(ki, _):
          slot = jax.lax.rem(ki, max_concurrent_steps)
          plgpu.barrier_wait(load_barriers.at[slot])  # Wait for data to arrive.
          plgpu.tcgen05_mma(
              acc_tmem,
              a_smem.at[slot],
              b_smem.at[slot],
              consumed_barriers.at[slot],
              accumulate=(ki > 0),
          )

        jax.lax.fori_loop(0, k_iters, _loop_body, None)
        plgpu.tcgen05_commit_arrive(mma_done_barrier)

    plgpu.barrier_wait(mma_done_barrier)
    acc_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(acc_smem, out_gmem.at[m_slice, n_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  f = plgpu.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=('m', 'n'),
      scratch_types=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k),
              dtype,
              transforms=transforms,
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, tile_n),
              dtype,
              transforms=transforms,
          ),
          acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
          acc_smem=plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
          load_barriers=plgpu.Barrier(
              num_arrivals=2, num_barriers=max_concurrent_steps
          ),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
          mma_done_barrier=plgpu.Barrier(
              num_arrivals=1, num_barriers=1, orders_tensor_core=True
          ),
      ),
  )
  return f(a, b)


def matmul2(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  if m % tile_m != 0:
    raise ValueError(f'{m=} must be divisible by {tile_m=}')
  if n % tile_n != 0:
    raise ValueError(f'{n=} must be divisible by {tile_n=}')
  if k % tile_k != 0:
    raise ValueError(f'{k=} must be divisible by {tile_k=}')
  m_iters = m // tile_m
  n_iters = n // tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(
      a_gmem,
      b_gmem,
      out_gmem,
      a_smem,
      b_smem,
      acc_tmem,
      acc_smem,
      load_barriers,
      consumed_barriers,
      mma_done_barrier,
  ):
    m_index = jax.lax.axis_index('m')
    n_index = jax.lax.axis_index('n')
    m_slice = pl.ds(m_index * tile_m, tile_m)
    n_slice = pl.ds(n_index * tile_n, tile_n)

    @plgpu.warp_map
    def _per_warp(warp_id):

      @pl.when(warp_id == 0)
      def _memory():
        def _loop_body(ki, _):
          slot = jax.lax.rem(ki, max_concurrent_steps)

          @pl.when(ki >= max_concurrent_steps)
          def _():  # Make sure the data has been consumed before overwriting.
            plgpu.barrier_wait(consumed_barriers.at[slot])

          k_slice = pl.ds(ki * tile_k, tile_k)
          plgpu.copy_gmem_to_smem(
              a_gmem.at[m_slice, k_slice],
              a_smem.at[slot],
              load_barriers.at[slot],
          )
          plgpu.copy_gmem_to_smem(
              b_gmem.at[k_slice, n_slice],
              b_smem.at[slot],
              load_barriers.at[slot],
          )

        jax.lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(warp_id == 1)
      def _compute():
        def _loop_body(ki, _):
          slot = jax.lax.rem(ki, max_concurrent_steps)
          plgpu.barrier_wait(load_barriers.at[slot])  # Wait for data to arrive.
          plgpu.tcgen05_mma(
              acc_tmem,
              a_smem.at[slot],
              b_smem.at[slot],
              consumed_barriers.at[slot],
              accumulate=(ki > 0),
          )

        jax.lax.fori_loop(0, k_iters, _loop_body, None)
        plgpu.tcgen05_commit_arrive(mma_done_barrier)

    plgpu.barrier_wait(mma_done_barrier)
    out_gmem_window = out_gmem.at[m_slice, n_slice]
    for ni in range(tile_n // config.epilogue_tile_n):
      acc_smem_ni = acc_smem.at[ni % 2]
      ni_slice = pl.ds(ni * config.epilogue_tile_n, config.epilogue_tile_n)
      # Make sure that previous copy is done before we overwrite.
      plgpu.wait_smem_to_gmem(1, wait_read_only=True)
      acc_smem_ni[...] = plgpu.async_load_tmem(acc_tmem.at[:, ni_slice]).astype(
          dtype
      )
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(acc_smem_ni, out_gmem_window.at[:, ni_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  f = plgpu.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=('m', 'n'),
      scratch_types=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k),
              dtype,
              transforms=transforms,
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, tile_n),
              dtype,
              transforms=transforms,
          ),
          acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
          acc_smem=plgpu.SMEM(
              (2, tile_m, config.epilogue_tile_n), dtype, transforms=transforms
          ),
          load_barriers=plgpu.Barrier(
              num_arrivals=2, num_barriers=max_concurrent_steps
          ),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
          mma_done_barrier=plgpu.Barrier(
              num_arrivals=1, num_barriers=1, orders_tensor_core=True
          ),
      ),
  )
  return f(a, b)


# TODO(nrink): Figure out how to safely run different instance of GPU
# interpret mode in parallel, and then remove this decorator.
@jtu.thread_unsafe_test_class()
class BlackwellExampleMatmulTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    mosaic_interpret.gpu_callbacks.reset_gpu_interpret_mode_state()

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    self.num_devices = jax.device_count()
    if self.num_devices > 1:
      self.skipTest(f'requires 1 device, found {self.num_devices}')

  @jtu.parameterized.product(
      detect_races=[False, True],
  )
  def test_matmul0(self, detect_races: bool):
    example_config = TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        max_concurrent_steps=4,
    )
    m, n, k = 256, 256, 256
    k1, k2 = jax.random.split(jax.random.key(0))
    a = jax.random.uniform(k1, (m, k), jnp.float16)
    b = jax.random.uniform(k2, (k, n), jnp.float16)

    device = jax.sharding.AbstractDevice(
        device_kind='NVIDIA B200',
        platform='gpu',
        num_cores=8,
    )
    with (
        jax.sharding.use_abstract_mesh(
            jax.sharding.AbstractMesh((), (), abstract_device=device)
        ),
        force_gpu_interpret_mode(InterpretParams(detect_races=detect_races)),
    ):
      res = matmul0(a, b, example_config).block_until_ready()

    expected = jnp.dot(a, b, preferred_element_type=jnp.float32)
    np.testing.assert_allclose(res, expected, rtol=1e-3)

  @jtu.parameterized.product(
      detect_races=[False, True],
  )
  def test_matmul1(self, detect_races: bool):
    example_config = TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        max_concurrent_steps=4,
    )
    m, n, k = 256, 256, 256
    k1, k2 = jax.random.split(jax.random.key(0))
    a = jax.random.uniform(k1, (m, k), jnp.float16)
    b = jax.random.uniform(k2, (k, n), jnp.float16)

    device = jax.sharding.AbstractDevice(
        device_kind='NVIDIA B200',
        platform='gpu',
        num_cores=8,
    )
    with (
        jax.sharding.use_abstract_mesh(
            jax.sharding.AbstractMesh((), (), abstract_device=device)
        ),
        force_gpu_interpret_mode(InterpretParams(detect_races=detect_races)),
    ):
      res = matmul1(a, b, example_config).block_until_ready()

    expected = jnp.dot(a, b, preferred_element_type=jnp.float32)
    np.testing.assert_allclose(res, expected, rtol=1e-3)

  @jtu.parameterized.product(
      detect_races=[False, True],
  )
  def test_matmul2(self, detect_races: bool):
    example_config = TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        max_concurrent_steps=4,
    )
    m, n, k = 256, 256, 256
    k1, k2 = jax.random.split(jax.random.key(0))
    a = jax.random.uniform(k1, (m, k), jnp.float16)
    b = jax.random.uniform(k2, (k, n), jnp.float16)

    device = jax.sharding.AbstractDevice(
        device_kind='NVIDIA B200',
        platform='gpu',
        num_cores=8,
    )
    with (
        jax.sharding.use_abstract_mesh(
            jax.sharding.AbstractMesh((), (), abstract_device=device)
        ),
        force_gpu_interpret_mode(InterpretParams(detect_races=detect_races)),
    ):
      res = matmul2(a, b, example_config).block_until_ready()

    expected = jnp.dot(a, b, preferred_element_type=jnp.float32)
    np.testing.assert_allclose(res, expected, rtol=1e-3)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
