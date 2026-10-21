# Copyright 2022 The JAX Authors.
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
import pickle
import sys
import threading

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


def _host_to_device_funcs():
  """Generates host-to-device transfer functions."""
  return [
      # (function name, is an explicit transfer?, function)
      ("host_to_device_jax_device_put", True,
       lambda: jax.device_put(np.ones(10))),
      ("host_to_device_jax_jit", False, lambda: jax.jit(lambda x: x)
       (np.ones(1))),
      ("host_to_device_jnp_one", False, lambda: jnp.ones(1)),
  ]


def _device_to_device_funcs():
  """Generates device-to-device transfer functions."""
  if len(jax.local_devices()) < 2:
    # device-to-device tests require at least 2 devices.
    return []

  with jax.transfer_guard_host_to_device("allow"):
    device_arrays = [jnp.ones(1) for _ in range(2)]
  return [
      # (function name, is an explicit transfer?, function)
      ("device_to_device_jax_device_put", True,
       lambda: jax.device_put(device_arrays[0], device=jax.local_devices()[1])),
      ("device_to_device_jax_jit", False,
       lambda: jax.jit(lambda x: x, device=jax.local_devices()[1])
       (device_arrays[1])),
  ]


def _device_to_host_funcs():
  """Generates device-to-host transfer functions.

  A CPU array in device memory reaches the host without a copy, and is held to
  the guard as an accelerator array's copy is (`array._guard_host_read`).
  """
  with jax.transfer_guard_host_to_device("allow"):
    device_arrays = [jnp.ones(1) for _ in range(6)]
    scalars = [jnp.ones(()) for _ in range(3)]
  funcs = [
      # (function name, is an explicit transfer?, function)
      ("device_to_host_jax_device_get", True,
       lambda: jax.device_get(device_arrays[0])),
      ("device_to_host_np_asarray", False,
       lambda: np.asarray(device_arrays[1])),
      ("device_to_host_copy_to_host_async", False,
       lambda: device_arrays[2].copy_to_host_async()),
      ("device_to_host_np_add", False, lambda: np.add(device_arrays[3], 1)),
      ("device_to_host_str", False, lambda: str(device_arrays[4])),
      ("device_to_host_pickle_dumps", False,
       lambda: pickle.dumps(device_arrays[5])),
      ("device_to_host_float", False, lambda: float(scalars[0])),
      ("device_to_host_int", False, lambda: int(scalars[1])),
      ("device_to_host_item", False, lambda: scalars[2].item()),
  ]
  if jax.default_backend() == "cpu" and sys.version_info >= (3, 12):
    # Only a CPU array has a host buffer to export.
    with jax.transfer_guard_host_to_device("allow"):
      exported = jnp.ones(1)
    funcs.append(("device_to_host_memoryview", False,
                  lambda: memoryview(exported)))
  return funcs


def _all_funcs():
  """Generates all transfer functions."""
  return (_host_to_device_funcs() + _device_to_device_funcs() +
          _device_to_host_funcs())


# List of test parameters shared by multiple tests.
_COMMON_TEST_PARAMETERS = [
    ("host_to_device", _host_to_device_funcs,
     jax.transfer_guard_host_to_device),
    ("device_to_device", _device_to_device_funcs,
     jax.transfer_guard_device_to_device),
    ("device_to_host", _device_to_host_funcs,
     jax.transfer_guard_device_to_host),
    ("all", _all_funcs, jax.transfer_guard),
]


# TransferGuardTest disables `--jax_enable_checks` because it
# can prematurely fetch the value of device arrays and make
# device-to-host tests to incur no transfers unexpectedly.
@jtu.with_config(jax_enable_checks=False)
class TransferGuardTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    # Nearly all test methods use the deprecated device argument to JIT.
    self.enter_context(jtu.ignore_warning(category=DeprecationWarning,
                                          message="backend and device argument"))

  @contextlib.contextmanager
  def assertAllows(self, func_name):
    """Asserts that a transfer in the context is allowed."""
    try:
      yield
    except Exception as e:
      raise RuntimeError(
          f"Expected a transfer to be allowed while running: {func_name}"
      ) from e

  @contextlib.contextmanager
  def assertLogs(self, func_name):
    """Asserts that a transfer in the context is logged and allowed."""
    # Only check if the transfer is allowed until Abseil provides an interface
    # to capture logs.
    with self.assertAllows(func_name):
      yield

  @contextlib.contextmanager
  def assertDisallows(self, func_name):
    """Asserts that a transfer in the context is disallowed."""
    try:
      with self.assertRaises(Exception):
        yield
    except Exception as e:
      raise RuntimeError(
          f"Expected a transfer to be disallowed while running: {func_name}"
      ) from e

  def test_simple(self):
    """Simple transfer guard tests."""
    with jax.transfer_guard("allow"):
      with self.assertAllows("host_to_device_jnp_ones"):
        jnp.ones(1)
    with jax.transfer_guard("log"):
      with self.assertLogs("host_to_device_jnp_ones"):
        jnp.ones(1)
    with jax.transfer_guard("disallow"):
      with self.assertDisallows("host_to_device_jnp_ones"):
        jnp.ones(1)

  def test_nesting(self):
    with jax.transfer_guard("disallow"):
      with jax.transfer_guard("allow"):
        with self.assertAllows("host_to_device_jnp_ones"):
          jnp.ones(1)
      with self.assertDisallows("host_to_device_jnp_ones"):
        jnp.ones(1)

  def test_mixed_nesting(self):
    with jax.transfer_guard_host_to_device("disallow"):
      with jax.transfer_guard("allow"):
        with self.assertAllows("host_to_device_jnp_ones"):
          jnp.ones(1)
      with self.assertDisallows("host_to_device_jnp_ones"):
        jnp.ones(1)

    with jax.transfer_guard("disallow"):
      with jax.transfer_guard_host_to_device("allow"):
        with self.assertAllows("host_to_device_jnp_ones"):
          jnp.ones(1)
      with self.assertDisallows("host_to_device_jnp_ones"):
        jnp.ones(1)

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_allow_by_default(self, func_generator, _):
    for func_name, _, func in func_generator():
      with self.assertAllows(func_name):
        func()

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_allow(self, func_generator, jax_transfer_guard):
    for func_name, _, func in func_generator():
      with jax_transfer_guard("allow"):
        with self.assertAllows(func_name):
          func()

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_log(self, func_generator, jax_transfer_guard):
    for func_name, explicit, func in func_generator():
      with jax_transfer_guard("log"):
        if explicit:
          with self.assertAllows(func_name):
            func()
        else:
          with self.assertLogs(func_name):
            func()

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_disallow(self, func_generator, jax_transfer_guard):
    for func_name, explicit, func in func_generator():
      with jax_transfer_guard("disallow"):
        if explicit:
          with self.assertAllows(func_name):
            func()
        else:
          with self.assertDisallows(func_name):
            func()

  @parameterized.named_parameters(
      ("device_to_host", _device_to_host_funcs,
       jax.transfer_guard_device_to_host),
      ("all", _device_to_host_funcs, jax.transfer_guard),
  )
  def test_disallow_ignores_arrays_on_cpu(self, func_generator,
                                          jax_transfer_guard):
    if jax.default_backend() == "cpu":
      self.skipTest("A CPU array in device memory keeps no host copy for a "
                    "later read to reuse; each read is held to the guard.")
    for func_name, _, func in func_generator():
      with jax_transfer_guard("allow"):
        # Transfer the device array to host.
        func()
      with jax_transfer_guard("disallow"):
        with self.assertAllows(func_name):
          # No error because the array has a value on host and no new transfer
          # will occur.
          func()

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_log_explicit(self, func_generator, jax_transfer_guard):
    for func_name, _, func in func_generator():
      with jax_transfer_guard("log_explicit"):
        with self.assertLogs(func_name):
          func()

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_disallow_explicit(self, func_generator, jax_transfer_guard):
    for func_name, _, func in func_generator():
      with jax_transfer_guard("disallow_explicit"):
        with self.assertDisallows(func_name):
          func()

  @jtu.run_on_devices("cpu")
  def test_disallow_allows_a_cpu_array_in_host_memory(self):
    with jax.transfer_guard_host_to_device("allow"):
      x = jnp.arange(4.)
    with jax.transfer_guard("disallow"):
      with self.assertDisallows("device_to_host_np_asarray"):
        np.asarray(x)
      for kind in ("pinned_host", "unpinned_host"):
        # An explicit move to host memory, as a checkpoint's copy makes.
        on_host = jax.device_put(x, x.sharding.with_memory_kind(kind))
        with self.assertAllows(f"device_to_host_np_asarray_{kind}"):
          np.testing.assert_array_equal(np.asarray(on_host), np.arange(4.))

  @jtu.thread_unsafe_test()  # Mutates global transfer guard config.
  def test_concurrent_global_state_tsan_race(self):
    stop = threading.Event()
    errors = []

    def config_writer():
      try:
        while not stop.is_set():
          jax.config.update("jax_transfer_guard_host_to_device", "log")
          jax.config.update("jax_transfer_guard_host_to_device", "allow")
      except Exception as e:
        errors.append(e)

    def reader():
      try:
        arr = np.ones(1)
        for _ in range(2000):
          jax.device_put(arr)
      except Exception as e:
        errors.append(e)

    threads = [
        threading.Thread(target=config_writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    try:
      for t in threads:
        t.start()
      threads[1].join()
      threads[2].join()
    finally:
      stop.set()
      for t in threads:
        t.join()
      jax.config.update("jax_transfer_guard_host_to_device", None)

    self.assertEmpty(errors)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
