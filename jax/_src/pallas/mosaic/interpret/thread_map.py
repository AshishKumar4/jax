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

import collections
from concurrent import futures
import contextlib
import dataclasses
import functools
import hashlib
import itertools
import threading
from typing import Any
import weakref

import jax
from jax._src import callback
from jax._src import core
from jax._src import source_info_util
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
    # The most recently returned first: a map after another runs on its
    # client, whose devices the bodies compiled for it name.
    client = _free_clients.pop(fitting[-1]) if fitting else None
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


# The executables compiled for thread maps' bodies, by what decides one (`_key`),
# most recently used last, the oldest dropped past `_COMPILED_HELD`. A key names
# a leased client's devices, which one map holds at a time, so no two maps ask
# for one key at once.
_COMPILED: collections.OrderedDict[tuple, Any] = collections.OrderedDict()
_COMPILED_HELD = 256
_compiled_lock = threading.Lock()


class _Identity:
  """A value keyed by its identity (one no key states by its contents: mutable,
  or unhashable), held so its identity is not reused while a key holds it."""
  __slots__ = ["value"]

  def __init__(self, value):
    self.value = value

  def __eq__(self, other):
    return isinstance(other, _Identity) and other.value is self.value

  def __hash__(self):
    return id(self.value)


def _structure(jaxpr: core.Jaxpr, *, consts: bool) -> tuple:
  """`jaxpr` as a key equal for two jaxprs that compute alike: its binders'
  types, its equations' primitives, operands by position, parameters (`_value`),
  contexts and effects, its outputs, and, where `consts`, its constants' values
  (a nested jaxpr's, which it computes with; a body's own are its executable's
  arguments). Its source locations and debug names are none of it."""
  numbered: dict[Any, int] = {}

  def bound(variables) -> tuple:
    for variable in variables:
      numbered[variable] = len(numbered)
    return tuple(_value(variable.aval) for variable in variables)

  def read(atom):
    if isinstance(atom, core.Literal):
      return ("literal", _value(atom.val), _value(atom.aval))
    return numbered[atom]

  binders = (bound(jaxpr.constvars), bound(jaxpr.invars))
  equations = tuple(
      (eqn.primitive, tuple(read(atom) for atom in eqn.invars),
       tuple(sorted((name, _value(param))
                    for name, param in eqn.params.items())),
       eqn.ctx, frozenset(eqn.effects), bound(eqn.outvars))
      for eqn in jaxpr.eqns)
  values = tuple(_value(const) for const in jaxpr.consts) if consts else ()
  return ("jaxpr", binders, values, equations,
          tuple(read(atom) for atom in jaxpr.outvars), frozenset(jaxpr.effects))


def _value(value):
  """A parameter's key: a jaxpr's structure with its constants, an array's
  bytes, a partial function's function and arguments, a frozen dataclass's
  fields (a host callback's function and argument tree), a source location as
  the place it names (it reports errors alone), a tuple's elements, a number
  by its type and bits (True, 1 and 1.0, -0.0 and 0.0, two NaNs' payloads,
  each equal as values, keyed apart), any other hashable immutable value
  itself (its own equality), else its identity
  (a list or a tracer among them)."""
  if isinstance(value, core.Jaxpr):
    return _structure(value, consts=True)
  if isinstance(value, core.Tracer):
    return _Identity(value)
  if isinstance(value, (np.ndarray, np.generic, jax.Array)):
    array = np.asarray(value)
    return ("array", array.shape, str(array.dtype),
            hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
  if isinstance(value, functools.partial):
    return ("partial", _value(value.func),
            tuple(_value(argument) for argument in value.args),
            tuple(sorted((name, _value(argument))
                         for name, argument in value.keywords.items())))
  params = getattr(type(value), "__dataclass_params__", None)
  if params is not None and params.frozen and not isinstance(value, type):
    return ("dataclass", type(value),
            tuple(_value(getattr(value, field.name))
                  for field in dataclasses.fields(value)))
  if isinstance(value, source_info_util.SourceInfo):
    return ("source", source_info_util.summarize(value))
  if isinstance(value, tuple):
    return ("tuple", tuple(_value(element) for element in value))
  if isinstance(value, float):
    return ("float", np.float64(value).tobytes())
  if isinstance(value, complex):
    return ("complex", np.complex128(value).tobytes())
  if isinstance(value, (bool, int)):
    return (type(value).__name__, value)
  if isinstance(value, (list, dict, set)):
    return _Identity(value)
  try:
    hash(value)
  except TypeError:
    return _Identity(value)
  return value


# Each body structure's id (`_structure`), the oldest dropped past
# `_STRUCTURES_HELD`; ids are never reused, so a dropped one only compiles again.
_STRUCTURES: collections.OrderedDict[tuple, int] = collections.OrderedDict()
_STRUCTURES_HELD = 1024
# Each body jaxpr object's structure id, found once an object: a kernel's thread
# map runs its body once a block, and walking its structure each time cost an
# interpreted kernel's test three to four times its run.
_BODIES: weakref.WeakKeyDictionary[core.Jaxpr, int] = (
    weakref.WeakKeyDictionary())


def _body(jaxpr) -> int:
  """`jaxpr`'s structure id, equal for two jaxprs of one structure."""
  with _compiled_lock:
    found = _BODIES.get(jaxpr)
  if found is not None:
    return found
  structure = _structure(jaxpr, consts=False)
  with _compiled_lock:
    found = _STRUCTURES.get(structure)
    if found is None:
      found = _STRUCTURES[structure] = next(_ids)
      if len(_STRUCTURES) > _STRUCTURES_HELD:
        _STRUCTURES.popitem(last=False)
    else:
      _STRUCTURES.move_to_end(structure)
    _BODIES[jaxpr] = found
  return found


_ids = itertools.count()


def _key(jaxpr, args) -> tuple:
  """What decides a thread map body's executable: its jaxpr's structure
  (`_body`) and its arguments' (its constants', its thread ID's and its
  inputs') types and devices, each device the object of its own client (two
  leased clients' first devices print alike). A kernel's forward, tangent and
  transpose traces each make a new jaxpr of one structure; keyed by the jaxpr
  object, each compiled again (a recurrence test's 76 bodies were 44 texts)."""
  types = tuple((np.shape(leaf), str(leaf.dtype), frozenset(leaf.devices()))
                for leaf in args)
  return _body(jaxpr), types


def _compiled(jaxpr, consts, args):
  """`jaxpr`'s executable for `consts` and `args`, compiled once a key."""
  key = _key(jaxpr, jax.tree.leaves((consts, args)))
  with _compiled_lock:
    executable = _COMPILED.get(key)
    if executable is not None:
      _COMPILED.move_to_end(key)
      return executable
  # Every argument kept, so a role body's compiled signature is its jaxpr's:
  # with its unused arguments pruned, a Blackwell causal forward's role thread
  # under Mosaic GPU's interpreter met XLA's verifier with a parameter tuple a
  # leaf short of its shardings (9 leaves against 10, hlo_sharding.cc:819).
  # Which pass drops the leaf is not established; keeping the arguments avoids
  # it, at the cost of passing buffers the body does not read.
  executable = jax.jit(_run, static_argnums=(0,), keep_unused=True).trace(
      jaxpr, consts, *args).lower().compile()
  with _compiled_lock:
    _COMPILED[key] = executable
    if len(_COMPILED) > _COMPILED_HELD:
      _COMPILED.popitem(last=False)
  return executable


def _thread_map_callback(jaxpr, token, device_id, num_threads, consts, invals,
                         *, on_exception):
  # Its inputs are the host's NumPy arrays as XLA hands them over
  # (`io_callback_numpy_inputs`), placed on the device once below: no copy to
  # the CPU device and back before that.
  device_id = int(device_id) if device_id is not None else None
  num_threads = int(num_threads)
  threads = []
  with _leased_devices(num_threads) as devices, futures.ThreadPoolExecutor(
      max_workers=num_threads
  ) as executor:
    # The constants and inputs go to each leased device once, and every thread
    # there reads those arrays rather than transferring them again (about 15%
    # of an interpreted kernel's time in one measured test).
    placed = [jax.device_put((consts, invals), d) for d in devices]
    # One executable a device, compiled before any thread runs: each thread
    # compiling its own raced the others to an empty cache, every one missing.
    try:
      executables = [
          _compiled(jaxpr, thread_consts,
                    jax.tree.leaves((jax.device_put(np.int32(0), device),
                                     *thread_invals)))
          for device, (thread_consts, thread_invals) in zip(devices, placed)]
    except Exception as e:
      on_exception(e, device_id=device_id, token=token)
      raise
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
          executor.submit(executables[device], thread_consts, *flat_args))
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
  # given jaxpr.  (JAX's caches should ensure the jaxpr is only lowered and
  # compiled once.)
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

  # Its constants go as operands, the jaxpr the callback holds bare: inside an
  # outer trace (a warp-specialized kernel's block body) they are that trace's
  # tracers, which a key holding the jaxpr would hold by identity, kept alive
  # and never matching another trace's.
  return _call_threadmap_callback(
      token, device_id, jaxpr.with_consts(()), num_threads, jaxpr.consts, args,
      use_ordered_callback, on_exception)
