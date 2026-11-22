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
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax._src.lib import xla_version
from jax._src.pallas.mosaic.interpret.thread_map import thread_map
from jax.experimental import io_callback
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

  def test_thread_map_in_program_dispatched_while_device_is_busy(self):
    if xla_version < 87:
      self.skipTest('Requires xla_version >= 87')

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


@jtu.thread_unsafe_test_class()
class InterpretThreadMapCompileTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

  def test_a_body_traced_again_compiles_once_for_all_its_threads(self):
    # Two thread maps of eight threads over one body, each traced anew (a
    # kernel's forward and transpose traces make new jaxprs of one text): one
    # compile, where each thread compiling its own raced the others to an empty
    # cache, and each new jaxpr compiled again.
    from jax._src import dispatch
    from jax._src import monitoring

    compiles = []

    def counted(event, duration, **kwargs):
      del duration
      if event == dispatch.BACKEND_COMPILE_EVENT and "_run" in kwargs.get(
          "fun_name", ""):
        compiles.append(kwargs["fun_name"])

    def f(thread, token, x):
      del token
      return jnp.cos(x * 1.25 + thread.astype(jnp.float32))

    x = jnp.arange(7, dtype=jnp.float32)
    monitoring.register_event_duration_secs_listener(counted)
    try:
      for _ in range(2):
        jax.block_until_ready(thread_map(f, 8, jnp.int32(0), x))
    finally:
      monitoring.unregister_event_duration_listener(counted)
    self.assertLen(compiles, 1, compiles)

  def test_a_body_run_again_is_keyed_without_walking_it_again(self):
    # An interpreted kernel's thread map runs its body once a block: the body's
    # structure is walked once a jaxpr object, not once a run.
    from jax._src.pallas.mosaic.interpret import thread_map as thread_map_lib

    walked = []
    structure = thread_map_lib._structure

    def counted(jaxpr, **kwargs):
      walked.append(jaxpr)
      return structure(jaxpr, **kwargs)

    def f(thread, token, x):
      del token
      return jnp.sin(x * 0.75 + thread.astype(jnp.float32))

    mapped = jax.jit(lambda x: thread_map(f, 4, jnp.int32(0), x))
    x = jnp.arange(5, dtype=jnp.float32)
    with mock.patch.object(thread_map_lib, "_structure", counted):
      for _ in range(3):
        jax.block_until_ready(mapped(x))
    self.assertLen(walked, 1)

  def test_bodies_alike_but_for_their_callbacks_each_run_their_own(self):
    # Two closures of one factory print alike in a jaxpr; each body still
    # calls its own.
    def recording(seen):
      def record(thread):
        seen.append(int(thread))
      return record

    def body(record):
      def f(thread, token):
        del token
        io_callback(record, None, thread)
      return f

    first, second = [], []
    for seen in (first, second):
      jax.block_until_ready(
          thread_map(body(recording(seen)), 2, jnp.int32(0)))
    self.assertEqual((sorted(first), sorted(second)), ([0, 1], [0, 1]))

  def test_bodies_alike_but_for_a_nested_constant_each_compute_their_own(self):
    # A nested jit's closed-over vector is its own jaxpr's constant, which a
    # printed jaxpr does not show; each body computes with its own.
    seen = []

    def record(value):
      seen.append(np.asarray(value).tolist())

    def body(vector):
      @jax.jit
      def shifted(thread):
        return thread.astype(jnp.float32) + vector

      def f(thread, token):
        del token
        io_callback(record, None, shifted(thread))
      return f

    for offset in (10.0, 20.0):
      jax.block_until_ready(thread_map(
          body(np.full((2,), offset, np.float32)), 2, jnp.int32(0)))
    self.assertCountEqual(seen, [[10.0, 10.0], [11.0, 11.0],
                                 [20.0, 20.0], [21.0, 21.0]])


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

    def meet():
      barrier.wait()

    def f(core_index, token):
      del core_index
      jax.experimental.io_callback(meet, (), ordered=True)
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
