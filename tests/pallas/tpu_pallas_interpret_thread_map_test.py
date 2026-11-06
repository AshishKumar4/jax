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

import threading
import time

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib import ifrt_version
from jax._src.pallas.mosaic.interpret.thread_map import thread_map
import jax.numpy as jnp

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

  def test_thread_map_in_program_dispatched_while_device_is_busy(self):
    if ifrt_version < 87:
      self.skipTest('Requires ifrt_version >= 87')

    # Programs with host callbacks, like `program`, run synchronously and take
    # no pacing units. `program` waits for `slow` on the dispatching thread and
    # returns before the 100 computations that use its output are dispatched,
    # so the computations of thread_map's threads never wait for their units.
    def f(core_index, token):
      # Uses `core_index`, so that the threads' computations have a live input
      # on the CPU, like interpret mode's.
      jax.experimental.io_callback(lambda _: None, (), core_index, ordered=True)
      return token

    program = jax.jit(lambda token: thread_map(f, 2, token))
    inc = jax.jit(lambda x: x + 1)

    @jax.jit
    def slow(a):
      for _ in range(16):
        a = jnp.tanh(a @ a)
      return a

    a = jnp.full((1024, 1024), 1e-4, jnp.float32)
    token = jnp.int32(0)
    queued = []

    def dispatch():
      # Compiles first: this thread's configs differ from the test's.
      jax.block_until_ready((slow(a), inc(program(token))))
      slow(a)  # Keeps the device busy, so that `program` has to wait for it.
      out = program(token)
      queued.extend(inc(out) for _ in range(100))

    # A daemon thread dispatches, so that the test can report a deadlock.
    threading.Thread(target=dispatch, daemon=True).start()
    deadline = time.monotonic() + 60
    while len(queued) < 100 or not all(x.is_ready() for x in queued):
      if time.monotonic() > deadline:
        self.fail('Computations did not finish within 60 s.')
      time.sleep(0.01)

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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
