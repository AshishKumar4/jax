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

from collections.abc import Hashable
from concurrent import futures
import contextlib
import dataclasses
import functools
import threading
from typing import Any

import jax
from jax._src import callback
from jax._src import core
from jax._src import util
from jax._src.lib import xla_client
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np


TOP_LEVEL_TOKEN_VALUE = 42
NESTED_TOKEN_VALUE = 1337


# The executions a CPU client's device holds in flight at once
# (cpu_client.h's LocalDeviceState, max_inflight_computations default): a
# leased client runs this many threads on each of its devices.
_THREADS_PER_DEVICE = 32

_free_clients: list[xla_client.Client] = []
_clients_lock = threading.Lock()


@contextlib.contextmanager
def _leased_devices(num_threads: int):
  """The devices of a CPU client of the interpreter's own, held by this thread
  map alone: one for every `_THREADS_PER_DEVICE` of its `num_threads` threads.

  A thread map runs inside a host callback of the computation that calls it,
  and runs its threads' programs from threads of its own: off the callback's
  thread, each execution takes a unit of its client's in-flight semaphore,
  which the callback's thread alone is exempt from (cpu_client.cc's
  ThisThreadIsInsideHostCallback). On the caller's client, computations queued
  behind the one running the callback can hold every unit, so the threads'
  programs wait for units that free only when the callback returns: a
  deadlock. On a client no other computation uses, the thread map's own
  executions are all that hold its units, at most `_THREADS_PER_DEVICE` on each
  of its devices (a cluster of blocks maps 36 threads, a test's six blocks of
  six)."""
  devices = -(-num_threads // _THREADS_PER_DEVICE)
  with _clients_lock:
    fitting = [i for i, held in enumerate(_free_clients)
               if len(held.local_devices()) >= devices]
    client = _free_clients.pop(fitting[0]) if fitting else None
  if client is None:
    client = xla_client.make_cpu_client(asynchronous=False, num_devices=devices)
  try:
    yield client.local_devices()[:devices]
  finally:
    with _clients_lock:
      _free_clients.append(client)


# top-level function so that caching works properly
def _run(jaxpr, consts, *args):
  jax_core.eval_jaxpr(jaxpr, consts, *args)


@dataclasses.dataclass(eq=False)
class _Programs:
  """The executable each distinct thread body compiles to, by its program's
  key (`_program_key`): a thread map traced again, as each call of a fresh
  `jit` or `pallas_call` traces its kernel anew and with it its thread bodies'
  jaxprs, runs the executable an identical one compiled, where jit's own cache,
  keyed by the jaxpr's identity, compiled it again at every launch."""

  compiled: dict[Hashable, Any] = dataclasses.field(default_factory=dict)
  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

  def cache_clear(self):
    with self.lock:
      self.compiled.clear()

  def cache_info(self):
    return len(self.compiled)

  def cache_keys(self):
    return list(self.compiled)


_programs = _Programs()
util.register_cache(_programs, "pallas interpret thread_map programs")


def _value_key(value):
  """`value` as it decides what a program computes: a callback by the function
  it calls and the values it is bound with (each trace binds a fresh
  `functools.partial`, equal to another by identity alone), a jaxpr by its
  equations' keys (`_jaxpr_key`), anything else as itself, by its own
  equality: an object equal to no other by value keys no other program."""
  if isinstance(value, (callback._FlatCallback, callback._NumpyInputs)):
    return type(value), tuple(_value_key(getattr(value, field.name))
                              for field in dataclasses.fields(value))
  if isinstance(value, functools.partial):
    return (_value_key(value.func), tuple(map(_value_key, value.args)),
            tuple(sorted((name, _value_key(each))
                         for name, each in value.keywords.items())))
  if isinstance(value, core.Jaxpr):
    return _jaxpr_key(value)
  if isinstance(value, (tuple, list)):
    return type(value), tuple(map(_value_key, value))
  return value


def _jaxpr_key(jaxpr: core.Jaxpr) -> Hashable:
  """The parameters of each of `jaxpr`'s equations, its sub-jaxprs' within,
  each by `_value_key`: what its printed form, which names each opaque
  parameter without its identity, does not decide."""
  return tuple((eqn.primitive, tuple(sorted(
      (name, _value_key(value)) for name, value in eqn.params.items())))
               for eqn in jaxpr.eqns)


def _program_key(jaxpr: core.Jaxpr) -> Hashable | None:
  """What a thread body's program computes, its operands apart: its jaxpr's
  printed form (its equations, their operands and literals, every aval) and
  its parameters (`_jaxpr_key`); None where a parameter has no hash."""
  key = (str(jaxpr), _jaxpr_key(jaxpr))
  try:
    hash(key)
  except TypeError:
    return None
  return key


def _run_jaxpr(jaxpr, key, consts, *args):
  # Every argument kept, so a role body's compiled signature is its jaxpr's:
  # with its unused arguments pruned, a Blackwell causal forward's role thread
  # under Mosaic GPU's interpreter met XLA's verifier with a parameter tuple a
  # leaf short of its shardings (9 leaves against 10, hlo_sharding.cc:819).
  # Which pass drops the leaf is not established; keeping the arguments avoids
  # it, at the cost of passing buffers the body does not read.
  def compiled():
    return jax.jit(_run, static_argnums=(0,), keep_unused=True).trace(
        jaxpr, consts, *args).lower().compile()

  if key is None:
    compiled()(consts, *args)
    return
  # Its operands as the executable takes them: their avals and their devices.
  key = key, tuple((jax.typeof(x), x.sharding)
                   for x in jax.tree.leaves((consts, args)))
  # Compiled under the lock, so a launch's threads compile their one program
  # once between them.
  with _programs.lock:
    program = _programs.compiled.get(key)
    if program is None:
      program = _programs.compiled[key] = compiled()
  program(consts, *args)
  return


def _thread_map_callback(jaxpr, token, device_id, num_threads, consts, invals,
                         *, on_exception):
  # Its inputs are the host's NumPy arrays as XLA hands them over
  # (`io_callback_numpy_inputs`), placed on the device once below: no copy to
  # the CPU device and back before that.
  device_id = int(device_id) if device_id is not None else None
  num_threads = int(num_threads)
  key = _program_key(jaxpr)
  threads = []
  with _leased_devices(num_threads) as devices, futures.ThreadPoolExecutor(
      max_workers=num_threads
  ) as executor:
    # The constants and inputs go to each leased device once, and every thread
    # there reads those arrays rather than transferring them again (about 15%
    # of an interpreted kernel's time in one measured test).
    placed = [jax.device_put((consts, invals), d) for d in devices]
    for i in range(num_threads):
      # `jaxpr` is the traced representation of a function whose first argument
      # is the thread ID. Hence,
      #   - prepend the thread ID onto the `invals`; and
      #   - flatten the arguments that are to be passed through to the
      #     evaluation of `jaxpr`.
      # The thread ID is placed there too: a body with no other operand runs
      # where its arguments are committed, which would otherwise be nowhere.
      device = i // _THREADS_PER_DEVICE
      thread_consts, thread_invals = placed[device]
      args = (jax.device_put(np.int32(i), devices[device]), *thread_invals)
      flat_args, _ = jax.tree.flatten(args)
      threads.append(
          executor.submit(_run_jaxpr, jaxpr, key, thread_consts, *flat_args))
    exceptions = []
    for i in range(num_threads):
      try:
        threads[i].result()
      except Exception as e:
        exceptions.append(e)
  if exceptions:
    on_exception(exceptions[0], device_id=device_id, token=token)
    # TODO(jburnim): Improve exception propagation here.  That is:
    #  - exceptions[0] might be an uninformative exception that just reports
    #    that the computation failed on a different device/core.
    #  - The cause of the exception (which may includes the actual line number
    #    in the user's kernel) is stripped by the XLA runtime.  (Maybe we could
    #    stash the exception in a global in Python somewhere.)
    raise exceptions[0]
  return token


def _call_threadmap_callback(token, device_id, jaxpr, num_threads, consts,
                             invals, use_ordered_callback, on_exception):
  # NOTE: At runtime, _thread_map_callback will lower and compile the
  # given jaxpr, once a process for each distinct program (`_programs`).
  #
  # TODO(jburnim): Would it be worth trying to lower/compile the jaxpr at
  # lowering/compilation time?  E.g., by using a custom primitive here, could
  # we lower/compile jaxpr at lowering time, and then pass the compiled
  # function to the callback?
  return callback.io_callback_numpy_inputs(
      functools.partial(_thread_map_callback, jaxpr, on_exception=on_exception),
      jax.ShapeDtypeStruct((), jnp.int32),
      token,
      device_id,
      num_threads,
      consts,
      invals,
      ordered=use_ordered_callback,
  )


def thread_map(
    f,
    num_threads,
    token,
    *args,
    use_ordered_callback=False,
    device_id=None,
    on_exception=lambda *args, **kwargs: None):
  """Executes `f(thread_id, token, *args)` for `num_threads` threads."""

  if num_threads == 1:
    # We're running `f` in the same JAX computation as the caller, so we thread
    # the token through.
    return f(jnp.int32(0), token, *args)

  def _f(core_or_thread_index, *args):
    # We are running `f` in sparate JAX computations on different threads from
    # the caller, so there cannot be any jaxpr-level dependencies/ordering
    # between IO callbacks in `f` and in the caller.  We pass a distinct value
    # (instead of the caller's `token`) to make this more clear.
    return f(core_or_thread_index, jnp.int32(NESTED_TOKEN_VALUE), *args)

  jaxpr = jax.make_jaxpr(_f)(jnp.int32(0), *args)

  return _call_threadmap_callback(
      token, device_id, jaxpr, num_threads, jaxpr.consts, args,
      use_ordered_callback, on_exception)
