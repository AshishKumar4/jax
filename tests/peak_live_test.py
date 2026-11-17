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

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib import xla_client as xc
import jax.experimental.topologies as topologies
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

jax.config.parse_flags_with_absl()


def _step(x, w):
  return jnp.tanh(x @ w) @ w


class PeakLiveTest(jtu.JaxTestCase):

  @jtu.run_on_devices("cpu", "cuda")
  def test_a_compiles_peak_holds_its_parameters_and_the_temporaries_it_reads_at_once(self):
    # tanh(x @ w) @ w: at the second product's peak, x's and w's buffers and
    # the first product's tanh, one temporary of a 256 x 256 f32 tile.
    tile = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    compiled = jax.jit(_step).lower(tile, tile).compile()
    live = compiled.peak_live()
    self.assertIsNotNone(live)
    parameters = [buffer for buffer in live if buffer.kind == "parameter"]
    self.assertLen(parameters, 2)
    for buffer in parameters:
      self.assertEqual((buffer.shape, buffer.dtype, buffer.bytes),
                       ((256, 256), "f32", 256 * 256 * 4))
    temporaries = [buffer for buffer in live if buffer.kind == "temporary"]
    self.assertTrue(any(
        buffer.shape == (256, 256) and buffer.dtype == "f32"
        and "tanh" in buffer.op_name for buffer in temporaries), live)
    self.assertLessEqual(sum(buffer.bytes for buffer in temporaries),
                         compiled.memory_analysis().temp_size_in_bytes)

  @jtu.run_on_devices("cuda")
  def test_a_compile_stopped_after_buffer_assignment_states_the_whole_compiles_peak(self):
    # For a compile-only GPU of the attached one's kind: the compile that stops
    # after buffer assignment, emitting no code, states the memory and the
    # buffers live at its peak as the whole compile does.
    target_config = xc.get_topology_for_devices(jax.devices()[:1]).target_config
    topology = topologies.get_topology_desc(
        platform="cuda", target_config=target_config, topology="1x1x1")
    mesh = topologies.make_mesh(topology, (1,), ("x",))
    tile = jax.ShapeDtypeStruct((256, 256), jnp.float32,
                                sharding=jax.sharding.NamedSharding(mesh, P()))
    lowered = jax.jit(_step).lower(tile, tile)
    whole = lowered.compile()
    stopped = lowered.compile({
        "xla_gpu_experimental_early_exit":
            "EARLY_EXIT_POINT_AFTER_BUFFER_ASSIGNMENT"})
    self.assertEqual(stopped.memory_analysis().temp_size_in_bytes,
                     whole.memory_analysis().temp_size_in_bytes)
    self.assertIsNotNone(whole.peak_live())
    self.assertEqual(stopped.peak_live(), whole.peak_live())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
