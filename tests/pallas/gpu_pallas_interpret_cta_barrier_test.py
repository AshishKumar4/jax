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

"""Named CTA barriers (`plgpu.CtaBarrier`) under the GPU kernel interpreter."""

import functools

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams as InterpretParams
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


@jtu.thread_unsafe_test_class()
class GpuPallasInterpretCtaBarrierTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("CPU-only test")
    try:
      # A previous test's exception from a jitted computation is consumed
      # here rather than propagated to this one.
      jax.effects_barrier()
    except:
      pass

  @jtu.parameterized.product(synchronized=[True, False])
  def test_a_hand_off_through_smem_waits_on_its_arrival(self, synchronized):
    """One warpgroup writes shared memory and arrives; the other arrives and
    waits, then reads it. Without the wait the race detector flags the read."""

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((128,), jnp.float32),
            barrier=plgpu.CtaBarrier(num_arrivals=2),
        ),
        interpret=InterpretParams(detect_races=True),
        num_threads=2,
        thread_name="wg",
    )
    def kernel(o_ref, smem_ref, barrier):
      wg = jax.lax.axis_index("wg")

      @pl.when(wg == 0)
      def _():
        smem_ref[...] = jnp.full((128,), 42.0, dtype=jnp.float32)
        plgpu.barrier_arrive(barrier)

      @pl.when(wg == 1)
      def _():
        if synchronized:
          plgpu.barrier_arrive_and_wait(barrier)
        o_ref[...] = smem_ref[...]
        if not synchronized:
          plgpu.barrier_arrive_and_wait(barrier)

    if synchronized:
      np.testing.assert_array_equal(kernel(), np.full((128,), 42.0, np.float32))
    else:
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, "RACE DETECTED"):
        kernel().block_until_ready()

  def test_a_barrier_of_an_array_is_its_own(self):
    """Each barrier of a `num_barriers` array keeps its own phase: a ping-pong
    over two, each warpgroup's turn its partner's arrival."""

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2, 128), jnp.float32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((128,), jnp.float32),
            turns=plgpu.CtaBarrier(num_arrivals=2, num_barriers=2),
        ),
        interpret=InterpretParams(detect_races=True),
        num_threads=2,
        thread_name="wg",
    )
    def kernel(o_ref, smem_ref, turns):
      wg = jax.lax.axis_index("wg")

      @pl.when(wg == 0)
      def _():
        smem_ref[...] = jnp.full((128,), 1.0, dtype=jnp.float32)
        plgpu.barrier_arrive(turns.at[0])
        plgpu.barrier_arrive_and_wait(turns.at[1])
        o_ref[0] = smem_ref[...]

      @pl.when(wg == 1)
      def _():
        plgpu.barrier_arrive_and_wait(turns.at[0])
        smem_ref[...] = smem_ref[...] + 1.0
        o_ref[1] = smem_ref[...]
        plgpu.barrier_arrive(turns.at[1])

    np.testing.assert_array_equal(
        kernel(), np.full((2, 128), 2.0, np.float32)
    )

  def test_a_wait_apart_from_its_arrival_is_refused(self):
    """A named barrier arrives and waits in one instruction (`bar.sync`): a
    wait alone has nothing to lower to."""

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_types=dict(barrier=plgpu.CtaBarrier(num_arrivals=1)),
        interpret=InterpretParams(),
    )
    def kernel(o_ref, barrier):
      plgpu.barrier_wait(barrier)
      o_ref[...] = jnp.zeros((128,), jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "no wait apart"):
      kernel()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
