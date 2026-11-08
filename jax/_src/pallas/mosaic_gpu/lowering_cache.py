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

"""A persistent cache of Pallas Mosaic GPU lowerings.

Lowering a kernel to a Mosaic GPU module takes about half a second, and a
program lowers the same kernels again in every process and every run. With
``jax_pallas_lowering_cache_dir`` set, ``lower_unpipelined_jaxpr_to_module``
keeps each module it lowers under a key, and reads it back for the same key.
Unset, it is ``lowering.lower_unpipelined_jaxpr_to_module`` and nothing else, and
nothing else of JAX is changed by this cache: it reads JAX, and never writes it.

The contract is the one of JAX's persistent compilation cache:

* *No stale hit in a process that has not changed JAX's internals.* The key is the
  identity of the installed code (``lowering_cache_seal``: every installed file of
  ``jax``, ``jaxlib`` and the CUDA plugins, hashed and compared with its wheel's
  ``RECORD``) and what the lowering is given, each part encoded explicitly and
  strictly (``lowering_cache_encoding``): the kernel's jaxpr with its params and
  avals, the rule's arguments, what the lowering reads of the process (the GPU, the
  shared memory a block may use, the device topology, NVSHMEM), every
  ``jax.config`` value and the contexts that are not flags, and the JAX, XLA,
  MOSAIC, NVSHMEM, CUDA and TRITON environment. A primitive, a class and a
  function are each named by the module that owns it, which the seal fixes.
* *What can be detected to differ from the installed code makes the lowering
  bypass the cache, counted as ``unsealed``:* an installed file that is not its
  wheel's, a missing ``RECORD`` (an editable or source install), a jax module
  loaded later that no ``RECORD`` lists, a rule registered in, or replaced in, a
  registry a lowering dispatches through (``lowering_cache_registries``) that is not
  sealed code, a registry that changes after the key was made, and a constant of
  the lowering's packages that is gone or holds what the key cannot hold
  (``lowering_cache_constants``).
* *A value the encoder cannot encode exactly bypasses the cache, counted as
  ``uncacheable``:* a class, enum, pytree node or aval no sealed module owns (a
  user's layout class), a primitive no sealed module holds, a mutable container.
  So does a lowering that profiles or dumps.
* *A process that changes JAX's internals is out of contract.* Rebinding a
  function or a method of JAX, replacing what a class or a module defines,
  writing into a rule's closure or a partial's keywords, or changing any of that
  from another thread while a kernel is lowered, is not detected: this cache does
  not instrument JAX to see it.

A cached lowering carries no source locations, whether it is a miss or a hit, so
a kernel moved in its file is still the same kernel.

Each boundary catches what it expects of it: the store's ``OSError``, an
entry's ``ValueError`` or ``KeyError``, MLIR's parse error, the encoder's
``Uncacheable``, the seal's ``Unsealed``. Anything else is an error.

``jax_pallas_lowering_cache_verify`` lowers every hit afresh as well and raises
when the two differ; a process that made a lowering through the cache with it
set and verified no hit exits with status 3. The counters are ``jax.monitoring``
events named
``/jax/pallas/mosaic_gpu/lowering_cache/{hit,miss,unsealed,uncacheable,verified}``
and the duration event ``.../seconds_saved``.
"""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import io
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Sequence
from typing import Any

import jax
from jax._src import config
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import layout as jax_layout
from jax._src import mesh as mesh_lib
from jax._src import monitoring
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.pallas.mosaic_gpu import lowering_cache_constants as constants_lib
from jax._src.pallas.mosaic_gpu import lowering_cache_encoding as encoding
from jax._src.pallas.mosaic_gpu import lowering_cache_registries as registries_lib
from jax._src.pallas.mosaic_gpu import lowering_cache_seal as sealing
from jax._src.pallas.mosaic_gpu import lowering_cache_store as store_lib
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.experimental.mosaic.gpu import lowering_inputs as mgpu_lowering_inputs
from jax.experimental.mosaic.gpu import utils as mgpu_utils
import ml_dtypes
import numpy as np

logger = logging.getLogger(__name__)

_EVENT_PREFIX = "/jax/pallas/mosaic_gpu/lowering_cache/"

# The cache's own controls: neither is part of a key. One policy for the
# jax.config sweep (by flag name) and the environment sweep (by variable name).
_CONTROLS = frozenset({
    "jax_pallas_lowering_cache_dir",
    "jax_pallas_lowering_cache_verify",
})

_ENVIRONMENT_PATTERN = re.compile(r"^(JAX|XLA|MOSAIC|NVSHMEM|CUDA|TRITON)")

# What no lowering reads: where a process keeps its compiled programs and
# which it keeps, and how its client allocates device memory. Each is a
# process's own (a test runner's share of one GPU, its job's cache
# directory), so none is part of a key, and processes alike in every input
# a lowering reads share their lowerings.
_UNREAD_FLAGS = frozenset({
    "jax_compilation_cache_dir",
    "jax_compilation_cache_max_size",
    "jax_compilation_cache_check_contents",
    "jax_compilation_cache_expect_pgle",
    "jax_compilation_cache_include_metadata_in_key",
    "jax_persistent_cache_min_compile_time_secs",
    "jax_persistent_cache_min_entry_size_bytes",
    "jax_persistent_cache_enable_xla_caches",
    "jax_raise_persistent_cache_errors",
    "jax_explain_cache_misses",
})
_UNREAD_VARIABLES = re.compile(
    r"^(XLA_PYTHON_CLIENT_|JAX_COMPILATION_CACHE_|JAX_PERSISTENT_CACHE_)")
# XLA_FLAGS' cache directories, each its job's own; every other flag a
# lowering may read (NVSHMEM's, `is_nvshmem_available`) stays keyed.
_UNREAD_XLA_FLAGS = re.compile(
    r"--xla_gpu_per_fusion_autotune_cache_dir=\S*|--xla_gpu_kernel_cache_file=\S*")


def _keyed_environment() -> dict[str, str]:
  """The environment variables a key holds: those naming a backend or a
  compiler, but not the cache's own controls nor what no lowering reads."""
  keyed = {}
  for name, value in os.environ.items():
    if (not _ENVIRONMENT_PATTERN.match(name) or name.lower() in _CONTROLS
        or _UNREAD_VARIABLES.match(name)):
      continue
    if name == "XLA_FLAGS":
      # Flags that are all unread are none: as if XLA_FLAGS were unset.
      value = " ".join(_UNREAD_XLA_FLAGS.sub("", value).split())
      if not value:
        continue
    keyed[name] = value
  return keyed
_DUMP_ENVIRONMENT_PREFIX = "MOSAIC_GPU_DUMP"
_DUMP_BASENAME = "mosaic_gpu.dump_basename"


class LoweringCacheMismatch(Exception):
  """A cached lowering differs from the same lowering made afresh."""


def _count(event: str) -> None:
  monitoring.record_event(_EVENT_PREFIX + event)
  if event == "verified":
    _verification.verified += 1


# What the verification of a process has done. A process that lowered through
# the cache with ``jax_pallas_lowering_cache_verify`` set, and verified no hit
# (a cold cache, an installation that is not sealed, a workload that is not
# cacheable), has verified nothing, and a run of it that is green would say it
# had: it exits with this status instead.
_EXIT_STATUS_UNVERIFIED = 3


class _Verification:

  def __init__(self):
    self.asked = False
    self.verified = 0
    self._lock = threading.Lock()

  def ask(self) -> None:
    """Notes that a lowering was made through the cache under verification."""
    with self._lock:
      if not self.asked:
        self.asked = True
        atexit.register(self.finish)

  def finish(self) -> None:
    if self.asked and not self.verified:
      sys.stdout.flush()
      sys.stderr.write(
          "jax_pallas_lowering_cache_verify was set and the Mosaic GPU lowering"
          " cache was used, but no hit was verified: exiting with status"
          f" {_EXIT_STATUS_UNVERIFIED}\n"
      )
      sys.stderr.flush()
      # An exit handler cannot change the status the interpreter exits with.
      os._exit(_EXIT_STATUS_UNVERIFIED)


_verification = _Verification()


_said: set[str] = set()


def _say_once(reason: str) -> None:
  """Tells, once a process, why lowerings are not cached: an unsealed one is
  otherwise silent (the `unsealed` counter says that, not why)."""
  if reason not in _said:
    _said.add(reason)
    logger.warning("the Mosaic GPU lowering cache is bypassed: %s", reason)


# ---------------------------------------------------------------------------
# The key.


# What reading the process's device can fail with: the GPU is one Mosaic does not
# support (a ValueError), there is no backend (a RuntimeError) or no device an
# IndexError. The lowering, which reads them again, raises what it raises today.
_INPUTS_UNAVAILABLE = (ValueError, RuntimeError, IndexError)


def _gather_inputs(jax_mesh: Any) -> mgpu_lowering_inputs.LoweringInputs:
  """What the lowering reads of the process, read once, outside the cache's context."""
  try:
    return mgpu_lowering_inputs.LoweringInputs(
        arch=tuple(int(a) for a in mgpu_utils._infer_arch()),
        max_smem_bytes=int(mgpu_core._max_smem_bytes()),
        single_process_multi_device_topology=bool(
            mgpu_core.is_single_process_multi_device_topology()
        ),
        nvshmem_available=bool(mgpu_core.is_nvshmem_available()),
        mesh=jax_mesh,
        known_multihost_mesh=bool(mgpu_core._is_known_multihost_mesh(jax_mesh)),
        smem_size_bound=mgpu_core._SMEM_SIZE_BOUND,
    )
  except _INPUTS_UNAVAILABLE as e:
    raise encoding.Uncacheable(f"a lowering input cannot be read: {e!r}") from e


def _eligible(params: gpu_core.CompilerParams) -> None:
  if params.profile_space:
    raise encoding.Uncacheable("a profiling lowering names a path of this process")
  for name in os.environ:
    if name.startswith(_DUMP_ENVIRONMENT_PREFIX):
      raise encoding.Uncacheable(f"{name} dumps this lowering")


def _key(
    ctx: mlir.LoweringRuleContext,
    gpu_mesh: gpu_core.Mesh,
    jax_mesh: Any,
    jaxpr: jax_core.Jaxpr,
    in_shapes: Sequence[Any],
    out_shapes: Sequence[Any],
    params: gpu_core.CompilerParams,
    inputs: mgpu_lowering_inputs.LoweringInputs,
    code: sealing.Seal,
    dispatch: bytes,
    constants: bytes,
) -> str:
  enc = encoding.Encoder()
  jaxpr_enc = enc.encode(jaxpr)
  registries_lib.check_primitives(enc.primitives)
  parts = {
      "version": b"mosaic-gpu-lowering-cache-key-v1",
      "installation": code.digest or b"",
      "what it dispatches through": dispatch,
      "constants": constants,
      "jaxpr": jaxpr_enc,
      # The rule's arguments.
      "input avals": enc.encode(tuple(in_shapes)),
      "output avals": enc.encode(tuple(out_shapes)),
      "compiler params": enc.encode(params),
      "gpu mesh": enc.encode(gpu_mesh),
      "jax mesh": enc.encode(jax_mesh),
      "axis environment": enc.encode(jax_core.get_axis_env()),
      "forward compatibility": enc.encode(bool(ctx.is_forward_compat())),
      "lowering parameters": enc.encode(ctx.module_context.lowering_parameters),
      # What the lowering reads of the process, fixed while it runs.
      "inputs": enc.encode((
          inputs.arch, inputs.max_smem_bytes,
          inputs.single_process_multi_device_topology,
          inputs.nvshmem_available, inputs.known_multihost_mesh,
          inputs.smem_size_bound,
      )),
      "abstract mesh": enc.encode(mesh_lib.get_abstract_mesh()),
      "concrete mesh": enc.encode(mesh_lib.get_concrete_mesh()),
      "physical mesh": enc.encode(mesh_lib.thread_resources.env.physical_mesh),
      "compute_on": enc.encode(config.compute_on_context_manager.value),
      "xla metadata": enc.encode(config.xla_metadata_context_manager.value),
      "no_collectives": enc.encode(config.no_collectives.value),
      "layout tracing mode": enc.encode(jax_layout.layout_tracing_mode.value),
      "interpreters": enc.encode((
          tuple(sys.version_info[:3]), sys.implementation.name, np.__version__,
          ml_dtypes.__version__,
      )),
      # Every jax.config value and the environment variables that name a
      # backend or a compiler, but not the cache's own controls nor what no
      # lowering reads (`_UNREAD_FLAGS`, `_keyed_environment`).
      "jax.config": enc.encode({
          name: value for name, value in config.config.values.items()
          if name.lower() not in _CONTROLS and name not in _UNREAD_FLAGS
      }),
      "environment": enc.encode(_keyed_environment()),
  }
  if logger.isEnabledFor(logging.DEBUG):
    logger.debug("the key's parts: %s", {
        name: hashlib.sha256(part).hexdigest()[:10] for name, part in parts.items()
    })
  return hashlib.sha256(
      b"".join(encoding.frame(b"k", p) for p in parts.values())
  ).hexdigest()


# ---------------------------------------------------------------------------
# A result and its stored form.


def _physical_dtype_name(dtype: Any) -> str:
  if isinstance(dtype, dtypes.ExtendedDType):
    dtype = jax_core.physical_element_aval(dtype).dtype
  return np.dtype(dtype).name


def _plain_shapes(shapes: Sequence[Any]) -> list[list[Any]]:
  return [
      [[int(d) for d in s.shape], _physical_dtype_name(s.dtype)] for s in shapes
  ]


def _shape_structs(plain: Sequence[Any]) -> tuple[jax.ShapeDtypeStruct, ...]:
  return tuple(
      jax.ShapeDtypeStruct(tuple(shape), np.dtype(name)) for shape, name in plain
  )


def _module_bytes(module: ir.Module) -> bytes:
  """The module's bytecode without its dump basename (a name of this process)."""
  attributes = module.operation.attributes
  saved = attributes[_DUMP_BASENAME] if _DUMP_BASENAME in attributes else None
  if saved is not None:
    del attributes[_DUMP_BASENAME]
  try:
    buffer = io.BytesIO()
    module.operation.write_bytecode(buffer)
    return buffer.getvalue()
  finally:
    if saved is not None:
      attributes[_DUMP_BASENAME] = saved


def _normalized(result: lowering.LoweringResult) -> store_lib.Entry:
  # A lowering that profiles is not cached (`_eligible`): its result has no
  # profiler to hold.
  assert result.profiler_spec is None
  return store_lib.Entry(
      module=_module_bytes(result.module),
      grid=tuple(int(g) for g in result.grid),
      block=tuple(int(b) for b in result.block),
      new_out_shapes=_plain_shapes(result.new_out_shapes),
      gmem_scratch_shapes=_plain_shapes(result.gmem_scratch_shapes),
      seconds=0.0,
  )


def _rebuild(
    entry: store_lib.Entry, context: ir.Context
) -> lowering.LoweringResult | None:
  """The lowering an entry holds, in `context`; none if its module does not parse."""
  mgpu.dialect.register_dialect(context)
  try:
    module = ir.Module.parse(entry.module, context=context)
  except ir.MLIRError as e:
    logger.warning("a Mosaic GPU lowering cache entry does not parse: %s", e)
    return None
  # As a fresh lowering's, the module takes its dump options of this process.
  mgpu.dialect.get_or_set_dump_options(module)
  return lowering.LoweringResult(
      module, entry.grid, entry.block,
      _shape_structs(entry.new_out_shapes), None,
      _shape_structs(entry.gmem_scratch_shapes),
  )


def _differences(hit: store_lib.Entry, fresh: store_lib.Entry) -> list[str]:
  return [
      name for name in (
          "module", "grid", "block", "new_out_shapes", "gmem_scratch_shapes"
      ) if getattr(hit, name) != getattr(fresh, name)
  ]


# ---------------------------------------------------------------------------
# The lowering.


def lower_unpipelined_jaxpr_to_module(
    ctx: mlir.LoweringRuleContext,
    gpu_mesh: gpu_core.Mesh,
    jax_mesh: Any,
    jaxpr: jax_core.Jaxpr,
    in_shapes: Sequence[jax_core.ShapedArray],
    out_shapes: Sequence[jax_core.ShapedArray],
    params: gpu_core.CompilerParams,
    *,
    outer_traceback: Any = None,
) -> lowering.LoweringResult:
  """``lowering.lower_unpipelined_jaxpr_to_module``, through the cache if it is on."""
  directory = config.jax_pallas_lowering_cache_dir.value

  def fresh() -> lowering.LoweringResult:
    return lowering.lower_unpipelined_jaxpr_to_module(
        gpu_mesh, jax_mesh, jaxpr, in_shapes, out_shapes, params,
        outer_traceback=outer_traceback,
    )

  if not directory:
    return fresh()
  if config.jax_pallas_lowering_cache_verify.value:
    _verification.ask()

  try:
    code = sealing.seal()
    if code.digest is None:
      raise sealing.Unsealed(code.reason)
    code.verify_loaded()
    _eligible(params)
    inputs = _gather_inputs(jax_mesh)
    dispatch = registries_lib.current()
    key = _key(
        ctx, gpu_mesh, jax_mesh, jaxpr, in_shapes, out_shapes, params, inputs,
        code, dispatch.digest, constants_lib.digest(),
    )
  except sealing.Unsealed as e:
    _say_once(str(e))
    _count("unsealed")
    return fresh()
  except encoding.Uncacheable as e:
    logger.debug("Mosaic GPU lowering is not cached (uncacheable): %s", e)
    _count("uncacheable")
    return fresh()

  context = ctx.module_context.context

  def lower_neutral() -> lowering.LoweringResult:
    # With no traceback to name a base location, an unknown ambient one, and
    # each equation's location unknown (`lowering_inputs.current()`), the
    # module holds no source location.
    with (
        mgpu_lowering_inputs.using(inputs),
        ir.Location.unknown(context),
    ):
      return lowering.lower_unpipelined_jaxpr_to_module(
          gpu_mesh, jax_mesh, jaxpr, in_shapes, out_shapes, params,
          outer_traceback=None,
      )

  store = store_lib.open_store(directory)  # None if the directory cannot be used.
  entry = None if store is None else store.get(key)
  if entry is not None:
    result = _rebuild(entry, context)
    if result is None:
      assert store is not None
      store.discard(key)
    else:
      if config.jax_pallas_lowering_cache_verify.value:
        lowered = lower_neutral()
        if not registries_lib.unchanged(dispatch):
          _count("unsealed")  # Not a lowering to compare the hit with.
          return lowered
        differences = _differences(_normalized(result), _normalized(lowered))
        if differences:
          raise LoweringCacheMismatch(
              f"the cached Mosaic GPU lowering {key} differs from a fresh one"
              f" in: {', '.join(differences)}"
          )
        _count("verified")
      _count("hit")
      monitoring.record_event_duration_secs(
          _EVENT_PREFIX + "seconds_saved", entry.seconds
      )
      return result

  started = time.perf_counter()
  result = lower_neutral()
  seconds = time.perf_counter() - started
  if not registries_lib.unchanged(dispatch):
    # A registry changed while the kernel was lowered (a rule that registers
    # another): the result is for this call, and is not the lowering the key names.
    _count("unsealed")
    return result
  _count("miss")
  if store is not None:
    try:
      store.put(key, dataclasses.replace(_normalized(result), seconds=seconds))
    except OSError as e:
      logger.warning("the Mosaic GPU lowering cache cannot be written: %s", e)
  return result
