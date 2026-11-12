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
"""What do real lowerings read, that a cache has to know of?

A cached lowering is right if everything it reads is in its key, or is what the
installed code defines (which the seal hashes), or cannot change what it makes.
This audit runs real lowerings (a simple kernel, a kernel of warps, the Hopper
matmul, attention, the Blackwell matmul), traces the code they execute, and looks
at every global that code loads (a module's name for a dict, a set, a config, a
context, an object): mutable state, which two processes that have not changed JAX
can still have differently. Each one is *covered* (the cache keys it, or checks
it) or *accepted*, with the reason, in `ACCEPTED`. One that is neither fails the
audit, so a table a lowering starts to read cannot go unnoticed.

It audits the paths these lowerings take, and no others.
"""

import builtins
import dis
import enum
import functools
import re
import sys
import threading
import types
import typing
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import core as jax_core
from jax._src import effects as jax_effects
from jax._src import layout as jax_layout
from jax._src import mesh as mesh_lib
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.pallas.mosaic_gpu import lowering_cache_constants as constants
from jax._src.pallas.mosaic_gpu import lowering_cache_registries as registries
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental import pallas as pl

def _covered_contexts() -> dict[int, str]:
  """The contexts and states the cache keys or checks, by the object (so that a
  module that re-exports one is covered as the module that defines it)."""
  from jax.experimental.mosaic.gpu import lowering_inputs  # pylint: disable=g-import-not-at-top
  return {
      id(config.compute_on_context_manager): "keyed",
      id(config.xla_metadata_context_manager): "keyed",
      id(config.no_collectives): "keyed",
      id(jax_layout.layout_tracing_mode): "keyed",
      id(mesh_lib.abstract_mesh_context_manager): "keyed (the abstract mesh)",
      id(mesh_lib.device_context): "keyed (the concrete mesh)",
      id(config.mesh_context_manager): "keyed (the physical mesh)",
      id(mesh_lib.thread_resources): "keyed (the physical mesh)",
      id(jax_core.top_axis_env): "keyed (the axis environment)",
      id(jax_effects.control_flow_allowed_effects):
          "keyed (its effect types, a table of the cache's identity)",
      id(lowering_inputs._current): "the cache's own: the answers the lowering reads",
  }


# Mutable state a lowering reads, and why the cache does not key it. By the name
# of the global: where an object is one (a sentinel, a singleton) a name says
# which, and a new global of another name is a new decision.
ACCEPTED = {
    # The trace the lowering sets for itself, and the contexts of tracing.
    "jax._src.core.trace_state": "the trace the lowering sets for the code it traces",
    "jax._src.core.trace_state_strong_ref": "the trace the lowering sets for the code it traces",
    "jax._src.core.trace_ctx": "the trace the lowering sets for the code it traces",
    "jax._src.core.eval_trace": "the top trace: a singleton",
    "jax._src.source_info_util._source_info_context":
        "source info and name stacks: locations, which a cached lowering has none of",
    "jax._src.test_util.thread_local_state": "the test utilities' own event listeners",
    # The devices: read once by the cache, and answered from what it read.
    "jax.local_devices": "devices: read by the cache once (lowering_inputs)",
    "jax._src.xla_bridge.local_devices": "devices: read by the cache once (lowering_inputs)",
    "jax._src.xla_bridge.get_backend": "devices: read by the cache once (lowering_inputs)",
    # Registries of the extensions: fixed by the code that registers at import.
    "jax._src.tree_util.default_registry":
        "pytree node types: the key holds the node types it flattens, sealed",
    "jax._src.tree_util.tracing_registry":
        "pytree node types: the key holds the node types it flattens, sealed",
    "jax._src.api_util.tracing_registry":
        "pytree node types: the key holds the node types it flattens, sealed",
    "jax._src.flattree.tracing_registry":
        "pytree node types: the key holds the node types it flattens, sealed",
    "jax._src.interpreters.mlir.upstream_dialects":
        "the dialects MLIR's upstream provides: registered by sealed code at import",
    "jaxlib._jax.config.unset": "a sentinel",
    # Names of things, not behaviour.
    "jax._src.lax.lax._JNP_FUNCTION_EQUIVALENTS": "names in an error message",
    "jax.experimental.mosaic.gpu.layout_inference.logger": "log output",
    "jax.experimental.mosaic.gpu.utils.logger": "log output",
    # Immutable objects of the installation: values, singletons and sentinels.
    "jax._src.config.no_default": "a sentinel",
    "jax._src.core.empty_mat": "an immutable value",
    "jax._src.mesh.empty_abstract_mesh": "an immutable value",
    "jax.experimental.mosaic.gpu.utils.HOPPER": "an immutable value (a frozen Arch)",
    "jax.experimental.mosaic.gpu.utils.BLACKWELL": "an immutable value (a frozen Arch)",
    "jax.experimental.mosaic.gpu.utils.RUBIN": "an immutable value (a frozen Arch)",
    "jax._src.core.internal_mutable_array_effect": "a stateless effect: the key holds effects by class",
    "jax._src.pallas.mosaic_gpu.core._memory_effect": "a stateless effect: the key holds effects by class",
    "jax._src.pallas.mosaic_gpu.core._pdl_effect": "a stateless effect: the key holds effects by class",
    "jax._src.pallas.mosaic_gpu.core._wgmma_pipeline_effect": "a stateless effect: the key holds effects by class",
    "jax._src.lax.utils.AutoLayout": "a singleton",
    "jax._src.linear_util.initial_result_paths": "a singleton",
    "jax._src.pjit.UNSPECIFIED": "a sentinel",
    "jax.experimental.mosaic.gpu.GLOBAL_BROADCAST": "a stateless marker",
    "jax.experimental.mosaic.gpu.launch_context.GLOBAL_BROADCAST": "a stateless marker",
    "jax.experimental.mosaic.gpu.dialect_lowering.RECURSED": "a sentinel",
    "jax.experimental.mosaic.gpu.launch_context._DYNAMIC_PEER_TABLE": "a sentinel",
    "jax._src.numpy.ufuncs.add": "a jnp.ufunc: an immutable wrapper of sealed functions",
    "jax._src.numpy.ufuncs.subtract": "a jnp.ufunc: an immutable wrapper of sealed functions",
}

# Tables that the installed code defines and fills at import, which no process
# that has not changed JAX has differently: the part of JAX's tracing that a
# lowering reads when it re-traces a jaxpr, and the dialect's own static tables. A
# library that registers an entry for its own type or primitive adds one no sealed
# kernel has (such a primitive makes its kernel uncacheable); replacing an entry of
# one the sealed code defines is changing JAX's internals, which is out of the
# cache's contract.
_DEFINED_BY_THE_INSTALLED_CODE = (
    "a table the installed code defines at import (a process that has not changed"
    " JAX's internals has it as it is; see the cache's contract)")
ACCEPTED.update({name: _DEFINED_BY_THE_INSTALLED_CODE for name in (
    "jax._src.config._jaxpr_eqn_ctx_configs",
    "jax._src.core.literalable_scalar_types",
    "jax._src.core.literalable_types",
    "jax._src.core.pytype_aval_mappings",
    "jax._src.dtypes._DEFAULT_TYPEMAP",
    "jax._src.dtypes._custom_float_dtypes",
    "jax._src.dtypes._custom_float_scalar_types",
    "jax._src.dtypes._dtype_to_32bit_dtype",
    "jax._src.dtypes._jax_dtype_set",
    "jax._src.dtypes._registered_weak_types",
    "jax._src.dtypes._weak_types",
    "jax._src.dtypes.python_scalar_types",
    "jax._src.dtypes.python_scalar_types_to_dtypes",
    "jax._src.interpreters.mlir._dtype_to_ir_type",
    "jax._src.interpreters.partial_eval.const_fold_rules",
    "jax._src.interpreters.partial_eval.custom_staging_rules",
    "jax._src.interpreters.partial_eval.forwarding_rules",
    "jax._src.interpreters.pxla.shard_arg_handlers",
    "jax._src.lax.lax._foldable_types",
    "jax._src.state.discharge._neutral_memory_spaces",
    "jax._src.state.indexing.indexer_transform_type_registry",
    "jax.experimental.mosaic.gpu.dialect_lowering.CMPF_IMPLS",
    "jax.experimental.mosaic.gpu.utils._MEMORY_SPACES",
)})

# By what a global is, besides what it is called.
_MEMOIZATION = "a cache of a function of its arguments (config, which is keyed, included)"
_LAYOUT = "a frozen layout the installed source defines (the seal's digest)"


def _short(value: Any) -> str:
  t = type(value)
  return f"{t.__module__}.{t.__qualname__}"


_SCALARS = (str, bytes, int, float, bool, complex, type(None), type(Ellipsis), range)
_IGNORED = (
    types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType,
    types.MethodDescriptorType, types.WrapperDescriptorType,
    types.GetSetDescriptorType, types.MemberDescriptorType, re.Pattern,
    enum.Enum, jax_core.Primitive, np.dtype, np.generic, property, staticmethod,
    classmethod, types.CodeType, type(threading.Lock()), type(threading.RLock()),
    types.UnionType, typing.TypeAliasType,
)


def _mutable_kind(value: Any) -> str | None:
  """What kind of state a global is, or None if it is not mutable state."""
  if isinstance(value, _SCALARS) or isinstance(value, _IGNORED):
    return None
  if type(value).__module__ == "typing":
    return None  # A type annotation (an alias, a literal): immutable.
  if isinstance(value, (tuple, frozenset)):
    return None if all(_mutable_kind(v) is None for v in value) else "container"
  if isinstance(value, functools.partial):
    return "partial"
  if isinstance(value, types.MethodType):
    return "method"
  return _short(value)


def _is_memoization(value: Any) -> bool:
  kind = _short(value)
  return "lru_cache" in kind or kind.startswith("jaxlib.weakref_lru_cache")


def _is_function_of_the_installation(value: Any) -> bool:
  """An extension function (its code is bytes RECORD hashes), or a jitted one
  (the Python function it wraps is sealed code)."""
  kind = _short(value)
  return kind in ("nanobind.nb_func", "jaxlib._jax.PjitFunction") or isinstance(
      value, types.BuiltinFunctionType)


def _covered(module: str, name: str, value: Any, table_ids: set[int],
             contexts: dict[int, str]) -> str | None:
  """Why the cache covers this global, or None."""
  if id(value) in table_ids:
    return "a registry of the cache's identity"
  if id(value) in contexts:
    return contexts[id(value)]
  if any(value is holder for holder in config.config._value_holders.values()):
    return "a config flag (jax.config.values)"
  if constants.in_scope(module) and isinstance(value, functools.partial):
    return "a partial the installed code defines (the seal's digest)"
  return None


def _accepted(module: str, name: str, value: Any) -> str | None:
  full = f"{module}.{name}"
  if full in ACCEPTED:
    return ACCEPTED[full]
  if _is_memoization(value):
    return _MEMOIZATION
  if isinstance(value, jax_core.AbstractValue):
    return "an immutable aval"
  if _short(value) == "builtins.NotImplementedType":
    return "a builtin singleton"
  if (constants.in_scope(module)
      and _short(value) == "jax.experimental.mosaic.gpu.fragmented_array.TiledLayout"):
    return _LAYOUT
  if _is_function_of_the_installation(value):
    return "a function of the installation (extension bytes, or the function a jit wraps)"
  return None


def _workloads():
  from jax.experimental.pallas.ops.gpu import attention_mgpu, blackwell_matmul_mgpu, hopper_matmul_mgpu  # pylint: disable=g-import-not-at-top
  attention_mgpu.cuda_versions = types.SimpleNamespace(
      cuda_runtime_get_version=lambda: 12090)

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((256,), jnp.float32), num_threads=1,
      thread_name="wg",
      scratch_types=[plgpu.SMEM((256,), jnp.float32), plgpu.Barrier()])
  def simple(x_ref, o_ref, smem, bar):
    plgpu.copy_gmem_to_smem(x_ref, smem, bar)
    plgpu.barrier_wait(bar)
    n = jax.lax.fori_loop(0, 3, lambda i, c: c + 1, 0)
    v = jax.lax.cond(n > 1, lambda: smem[...] * 3.0, lambda: smem[...])
    smem[...] = v + jnp.where(v > 0, v, 0.0)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(smem, o_ref)
    plgpu.wait_smem_to_gmem(0)

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((4, 2, 64), jnp.float32),
      scratch_types=(plgpu.SMEM((2, 4, 64), jnp.float32),))
  def warps(in_ref, out_ref, smem):
    smem[...] = in_ref[...] * 2.0

    @plgpu.warp_map
    def _(warp_id):
      @pl.when(warp_id == 0)
      def _():
        plgpu.copy_smem_to_gmem(smem.transpose((1, 0, 2)), out_ref)
        plgpu.wait_smem_to_gmem(0)

  def hopper(a, b):
    return hopper_matmul_mgpu.matmul(a, b, None, hopper_matmul_mgpu.TuningConfig(
        tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4))

  def attention(q, k, v):
    return attention_mgpu.attention(q, k, v, attention_mgpu.TuningConfig(
        block_q=128, block_kv=128, max_concurrent_steps=2))

  def blackwell(a, b):
    return blackwell_matmul_mgpu.matmul_kernel(
        a, b, blackwell_matmul_mgpu.TuningConfig(
            tile_m=128, tile_n=128, tile_k=128, max_concurrent_steps=2,
            collective=False))

  bf = lambda *s: jnp.zeros(s, jnp.bfloat16)
  fp = lambda *s: jnp.zeros(s, jnp.float16)
  return [
      ("simple", simple, (jnp.ones((256,), jnp.float32),), "9.0"),
      ("warps", warps, (jnp.ones((2, 4, 64), jnp.float32),), "9.0"),
      ("hopper matmul", hopper, (bf(1024, 1024), bf(1024, 1024)), "9.0"),
      ("attention", attention, tuple(fp(2, 512, 4, 128) for _ in range(3)), "9.0"),
      ("blackwell matmul", blackwell, (bf(1024, 1024), bf(1024, 1024)), "10.0"),
  ]


def run() -> dict[str, Any]:
  """Lowers the workloads, traced; what each global a lowering loads is."""
  executed: dict[types.CodeType, dict[str, Any]] = {}

  def profile(frame, event, arg):
    del arg
    if event == "call" and frame.f_code not in executed:
      executed[frame.f_code] = frame.f_globals

  original = lowering.lower_unpipelined_jaxpr_to_module

  def traced(*args, **kwargs):
    threading.setprofile(profile)
    sys.setprofile(profile)
    try:
      return original(*args, **kwargs)
    finally:
      sys.setprofile(None)
      threading.setprofile(None)

  lowering.lower_unpipelined_jaxpr_to_module = traced
  lowered = {}
  try:
    for name, f, args, capability in _workloads():
      jax.clear_caches()
      device = jax.sharding.AbstractDevice(
          device_kind="NVIDIA GPU", platform="gpu", num_cores=132,
          compute_capability=capability)
      mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
      with jax.sharding.use_abstract_mesh(mesh):
        jax.jit(f).trace(*args).lower(lowering_platforms=("cuda",))
      lowered[name] = True
  finally:
    lowering.lower_unpipelined_jaxpr_to_module = original

  # (`mosaic_lowering_rules` holds the per-semantics registries the key holds.)
  table_ids = {id(t.get()) for t in registries.tables()} | {
      id(lowering.mosaic_lowering_rules)}
  contexts = _covered_contexts()
  found: dict[tuple[str, str], Any] = {}
  for code, namespace in executed.items():
    module = namespace.get("__name__", "?")
    if not module.startswith(("jax", "jaxlib")):
      continue
    previous = None
    for instruction in dis.get_instructions(code):
      if instruction.opname == "LOAD_GLOBAL":
        name = instruction.argval
        value = namespace.get(name, getattr(builtins, name, None))
        if _mutable_kind(value) is not None:
          found[(module, name)] = value
      elif (instruction.opname in ("LOAD_ATTR", "LOAD_METHOD")
            and previous is not None and previous.opname == "LOAD_GLOBAL"):
        base = namespace.get(previous.argval)
        if isinstance(base, types.ModuleType) and base.__name__.startswith(("jax", "jaxlib")):
          value = getattr(base, instruction.argval, None)
          if _mutable_kind(value) is not None:
            found[(base.__name__, instruction.argval)] = value
      previous = instruction

  covered, accepted, uncovered = {}, {}, {}
  for (module, name), value in sorted(found.items()):
    why = _covered(module, name, value, table_ids, contexts)
    if why is not None:
      covered[f"{module}.{name}"] = why
      continue
    why = _accepted(module, name, value)
    if why is not None:
      accepted[f"{module}.{name}"] = why
      continue
    uncovered[f"{module}.{name}"] = _mutable_kind(value)
  return {
      "lowered": sorted(lowered), "executed_code_objects": len(executed),
      "covered": len(covered), "accepted": accepted, "uncovered": uncovered,
  }
