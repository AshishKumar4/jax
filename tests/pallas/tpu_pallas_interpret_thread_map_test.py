# Copyright 2024 The JAX Authors.
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

"""Thread map test for TPU-specific interpret mode."""

import subprocess
import sys
import textwrap
import threading

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic.interpret.thread_map import thread_map
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


# TODO(jburnim): Figure out how to safely run different instance of TPU
# interpret mode in parallel, and then remove this decorator.
@jtu.thread_unsafe_test_class()
class InterpretThreadMapTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    self.num_devices = jax.device_count()
    if self.num_devices > 1:
      # Workaround for https://github.com/jax-ml/jax/issues/25671
      self.skipTest(f'requires 1 device, found {self.num_devices}')

  def test_thread_map(self):
    barrier = threading.Barrier(8)
    lock = threading.Lock()
    concurrent_calls = [0]
    max_concurrent_calls = [0]

    def _barrier():
      with lock:
        concurrent_calls[0] += 1
        max_concurrent_calls[0] = max(
            max_concurrent_calls[0], concurrent_calls[0])
      barrier.wait()
      with lock:
        concurrent_calls[0] -= 1

    def f(core_index, token):
      del core_index
      jax.experimental.io_callback(_barrier, (), ordered=True)
      return token

    thread_map(f, 8, jnp.int32(0))
    self.assertEqual(max_concurrent_calls[0], 8)
    # `thread_map` returns only after all threads have completed, so the final
    # value of `concurrent_calls` should be zero.
    self.assertEqual(concurrent_calls[0], 0)

  def test_thread_map_reads_its_inputs_back_from_no_device(self):
    # Its constants and inputs reach its threads as the host's arrays, put on
    # the device once: none is read back from the CPU device first, which the
    # device-to-host guard would hold.
    seen = []

    def f(core_index, token, x):
      jax.experimental.io_callback(seen.append, (), x[0] + core_index,
                                   ordered=True)
      return token

    x = jnp.arange(3, dtype=jnp.int32)
    with jax.transfer_guard_device_to_host('disallow'):
      jax.block_until_ready(thread_map(f, 4, jnp.int32(0), x))
    self.assertCountEqual([int(value) for value in seen], [0, 1, 2, 3])

  def test_thread_map_keeps_unused_args_beside_device_inputs(self):
    # Each thread's body runs with every argument it was traced with, those
    # it does not read between those it does, beside a captured device
    # constant: each thread reads its live inputs, whatever the unused ones.
    seen = []
    scale = jnp.arange(4, dtype=jnp.float32) * 3

    def f(core_index, token, x, unused, y):
      del unused
      jax.experimental.io_callback(
          seen.append, (), x[0] + y[1] + scale[1] + core_index.astype(jnp.float32),
          ordered=True)
      return token

    x = jax.device_put(jnp.arange(3, dtype=jnp.float32), jax.devices()[0])
    unused = jnp.ones((5,), jnp.float32)
    y = jnp.full((2,), 10.0, jnp.float32)
    jax.block_until_ready(thread_map(f, 4, jnp.int32(0), x, unused, y))
    self.assertCountEqual([float(value) for value in seen],
                          [13.0, 14.0, 15.0, 16.0])


# A process whose CPU client holds every unit of its in-flight semaphore (32,
# cpu_client.h) in computations that each run a host callback, every callback
# waiting at a barrier until all hold theirs, each callback then running a
# thread map. A callback-only program counts no FLOPs, so the client runs each
# inline on its caller's thread, which holds its unit until it returns.
_SATURATED = textwrap.dedent("""
    import concurrent.futures, sys, threading
    import jax, numpy as np
    from jax.experimental import io_callback
    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import thread_map

    count, inputs = int(sys.argv[1]), int(sys.argv[2])
    entered = threading.Barrier(count + 1)
    if inputs:
      body = jax.make_jaxpr(lambda thread, x: jnp.sin(x))(np.int32(0), np.float32(0))
    else:
      body = jax.make_jaxpr(lambda thread: jnp.sin(thread * 1.0))(np.int32(0))

    def callback(x):
      entered.wait()
      thread_map._thread_map_callback(
          body.jaxpr, np.int32(42), None, np.int32(2), body.consts,
          (x,) if inputs else (), on_exception=lambda *args, **kwargs: None)
      return np.float32(x + 1)

    launch = jax.jit(lambda x: io_callback(
        callback, jax.ShapeDtypeStruct((), np.float32), x)).lower(
            np.float32(0)).compile()
    with concurrent.futures.ThreadPoolExecutor(count) as drivers:
      pending = [drivers.submit(launch, np.float32(0)) for _ in range(count)]
      entered.wait()
      for future in pending:
        np.testing.assert_array_equal(np.asarray(future.result()), 1)
    print("done")
""")


@jtu.thread_unsafe_test_class()
class InterpretThreadMapSaturationTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

  @parameterized.parameters((31, 1), (32, 1), (32, 0))
  def test_thread_map_runs_inside_callbacks_holding_every_unit(
      self, count, inputs):
    # Each callback's thread map runs its threads' programs off the callback's
    # thread. With every unit of the caller's client held by the computations
    # running those callbacks (32), a thread's program waiting for a unit there
    # would wait for ever; one fewer (31) leaves it one. A thread map runs its
    # threads on a client of its own, so all finish: a body with no input
    # besides its thread ID too.
    try:
      ran = subprocess.run(
          [sys.executable, "-c", _SATURATED, str(count), str(inputs)],
          capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
      self.fail(f"{count} saturating callbacks' thread maps never returned")
    self.assertEqual(ran.returncode, 0, ran.stderr[-2000:])
    self.assertIn("done", ran.stdout)

  def test_a_thread_map_wider_than_a_devices_executions_runs_all_at_once(self):
    # 40 threads, more than one device holds in flight (32), all meet at one
    # barrier: every thread's program is in flight at once.
    barrier = threading.Barrier(40)

    def f(core_index, token):
      del core_index
      jax.experimental.io_callback(barrier.wait, (), ordered=True)
      return token

    jax.block_until_ready(thread_map(f, 40, jnp.int32(0)))
    self.assertEqual(barrier.n_waiting, 0)

  def test_thread_map_computes_as_its_callers_client_does(self):
    # A thread's program, run on the thread map's own client, gives what the
    # caller's client gives bit for bit: a reduction XLA:CPU may split across
    # its intra-op threads included.
    x = np.random.default_rng(0).standard_normal((257, 1031)).astype(np.float32)
    seen = []

    def f(thread, token, x):
      del thread
      jax.experimental.io_callback(
          seen.append, (), jnp.sum(jnp.exp(x) * x, axis=1), ordered=True)
      return token

    jax.block_until_ready(thread_map(f, 2, jnp.int32(0), x))
    expected = np.asarray(jax.jit(lambda x: jnp.sum(jnp.exp(x) * x, axis=1))(x))
    self.assertLen(seen, 2)
    for got in seen:
      np.testing.assert_array_equal(np.asarray(got), expected)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
