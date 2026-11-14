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
"""One process of the Mosaic GPU lowering cache tests.

Run as ``python -m mosaic_gpu_lowering_cache_support.child``, with its case as JSON
in ``LOWERING_CACHE_CASE``. It lowers a kernel for a compile-only GPU as the case
says, through the cache the environment names, and prints one ``RESULT`` line: what
the cache counted and what the lowering produced, or what raised.

A case's keys (all optional but `workdir`): `kernel` (a function of `kernels`),
`kernel_args`, `size`, `rows`, `shape`, `power`, `scale`, `pad` (lines added above
the kernels' file), `capability` (the GPU the mesh states), `compiler_params`,
`site` (the sealed site, whose manifests seal the process), `config` (jax.config
values), `smem_bound`, `nvshmem_bc` (what the process answers of NVSHMEM), `mesh`
(a mesh of two devices: "abstract2" or "concrete2"), `repeat`, `tweaks` (changes
to the process made before the first lowering), `after` (made between the first and
a second), `mutate` and `touch` (changes to the registries), `late_module` (a
module imported between the two), `bug` (one of the cache's own), `direct` (the
lowering with no cache around it), `report_seal`, and `mode` ("encoder": the
canonicalizer's cases; "audit": what real lowerings read).
"""

import hashlib
import importlib
import importlib.util
import io
import json
import os
import pathlib
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from jax import monitoring
from jax._src import core as jax_core
from jax._src.lax import eval_jaxpr as eval_jaxpr_lib
from jax._src.lib.mlir import ir
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.pallas.mosaic_gpu import lowering_cache
from jax._src.pallas.mosaic_gpu import lowering_cache_encoding as cache_encoding
from jax._src.pallas.mosaic_gpu import lowering_cache_seal as cache_seal
from jax._src.state import discharge
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.experimental.mosaic.gpu import dialect_lowering
from jax.experimental.mosaic.gpu import layout_inference
from jax.experimental.mosaic.gpu import lowering_inputs
from jax.experimental.pallas import mosaic_gpu as plgpu

_PREFIX = "/jax/pallas/mosaic_gpu/lowering_cache/"
_DUMP_BASENAME = "mosaic_gpu.dump_basename"
_SUPPORT = pathlib.Path(__file__).resolve().parent


class Recorder:
  """What happened in a lowering: the cache's events, the lowering it made, and
  what it read of the device topology.

  It watches without changing the lowering's own code (a replaced function of
  a lowering package is what the cache refuses): the results are what the
  cache's wrapper returns, which is wrapped here, and the device reads are
  those of `jax`'s public functions.
  """

  def __init__(self):
    self.events: list[str] = []
    self.saved: list[float] = []
    self.results: list[lowering.LoweringResult] = []
    # (name, whether the read was made while the cache's snapshot answered).
    self.reads: list[tuple[str, bool]] = []
    # What the pallas_call's lowering calls: the cache's wrapper, or (a case that
    # asks for it) the lowering with no cache around it.
    self.inner = lowering_cache.lower_unpipelined_jaxpr_to_module

  def install(self):
    monitoring.register_event_listener(
        lambda e, **kw: self.events.append(e[len(_PREFIX):])
        if e.startswith(_PREFIX) else None)
    monitoring.register_event_duration_secs_listener(
        lambda e, d, **kw: self.saved.append(d)
        if e == _PREFIX + "seconds_saved" else None)

    def capture(*args, **kwargs):
      result = self.inner(*args, **kwargs)
      self.results.append(result)
      return result

    lowering_cache.lower_unpipelined_jaxpr_to_module = capture

  def reset(self):
    for items in (self.events, self.saved, self.results, self.reads):
      del items[:]

  def install_topology_spies(self):
    """Watches the device reads the lowering makes, without changing any function
    (a replaced function of jax is what the cache refuses): each call of one of
    them is an event of the interpreter's monitoring, for the code of that function."""
    from jax.extend import backend as jex_backend  # pylint: disable=g-import-not-at-top
    tool = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(tool, "lowering-cache-test")
    names = {}
    for owner, name in ((jax, "device_count"), (jax, "local_device_count"),
                        (jax, "local_devices"),
                        (jex_backend, "get_default_device")):
      code = getattr(getattr(owner, name), "__code__", None)
      if code is not None:
        names[code] = name

    def started(code, offset):
      del offset
      self.reads.append((names[code], lowering_inputs.current() is not None))

    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
    for code in names:
      sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)


def normalize(result):
  """The result as a cache keeps it: the module without its dump basename, a
  name of the process, and the rest as plain data."""
  attributes = result.module.operation.attributes
  saved_attribute = attributes[_DUMP_BASENAME] if _DUMP_BASENAME in attributes else None
  if saved_attribute is not None:
    del attributes[_DUMP_BASENAME]
  buffer = io.BytesIO()
  result.module.operation.write_bytecode(buffer)
  # The module as text, which holds no locations: comparable between a lowering
  # with them and one without.
  asm = result.module.operation.get_asm()
  if saved_attribute is not None:
    attributes[_DUMP_BASENAME] = saved_attribute
  data = buffer.getvalue()
  shapes = lambda ss: [[list(s.shape), np.dtype(s.dtype).name] for s in ss]
  return {
      "module": hashlib.sha256(data).hexdigest(),
      "module_size": len(data),
      "grid": list(result.grid),
      "block": list(result.block),
      "new_out_shapes": shapes(result.new_out_shapes),
      "gmem_scratch_shapes": shapes(result.gmem_scratch_shapes),
      "profiler_spec_is_none": result.profiler_spec is None,
      "asm": hashlib.sha256(asm.encode()).hexdigest(),
  }


# -- kernels

def load_kernels(case):
  """The kernels' file, copied to the work directory with `pad` lines above it.

  Processes that lower together share the work directory and the file: each
  writes its copy whole beside it and renames it into place, so none reads a
  copy another is still writing.
  """
  path = pathlib.Path(case["workdir"]) / f"kernels_{case['pad']}.py"
  written = path.with_name(f"{path.name}.{os.getpid()}")
  written.write_text("\n" * case["pad"] + (_SUPPORT / "kernels.py").read_text())
  os.replace(written, path)
  spec = importlib.util.spec_from_file_location(path.stem, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def make_kernel(case):
  params = case.get("compiler_params")
  if params is not None:
    params = plgpu.CompilerParams(**{
        name: value.encode() if name == "metadata" else value
        for name, value in params.items()})
  factory = getattr(load_kernels(case), case.get("kernel", "plain"))
  return factory(case["size"], case["power"], params, case["scale"],
                 **case.get("kernel_args", {}))


def make_input(case):
  shape = tuple(case["shape"]) if case.get("shape") else (
      (case["rows"], case["size"]) if case.get("rows") else (case["size"],))
  return jnp.ones(shape, jnp.float32)


# -- lowering

def lower(case, kernel, x):
  jax.clear_caches()
  device = jax.sharding.AbstractDevice(
      device_kind="NVIDIA H100 80GB HBM3", platform="gpu", num_cores=132,
      compute_capability=case["capability"])
  kind = case.get("mesh")
  if kind is None:
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    with jax.sharding.use_abstract_mesh(mesh):
      return jax.jit(kernel).trace(x).lower(lowering_platforms=("cuda",))
  # A kernel run by each of two devices, of a mesh that is a compile-only
  # target's (an abstract one) or the process's own (its two CPUs), lowered for
  # the GPU.
  if kind == "abstract2":
    mesh = jax.sharding.AbstractMesh(
        (2,), ("x",), (jax.sharding.AxisType.Auto,), abstract_device=device)
  else:
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("x",))
  p = jax.sharding.PartitionSpec
  body = jax.shard_map(kernel, mesh=mesh, in_specs=p(), out_specs=p(), check_vma=False)
  return jax.jit(body).trace(x).lower(lowering_platforms=("cuda",))


def caller_metadata(lowered) -> str | None:
  """What the lowered program's Mosaic GPU call carries of its caller
  (`CompilerParams.metadata`), read from the call's backend config."""
  found = []

  def visit(operation):
    if operation.name == "stablehlo.custom_call":
      config = operation.attributes["backend_config"]
      if ir.DictAttr.isinstance(config) and "metadata" in (config := ir.DictAttr(config)):
        found.append(ir.StringAttr(config["metadata"]).value)
    return ir.WalkResult.ADVANCE

  lowered.compiler_ir("stablehlo").operation.walk(visit)
  return found[0] if found else None


def one(case, rec, kernel, x):
  """One lowering: what the cache counted, and what it produced or raised."""
  rec.reset()
  try:
    lowered = lower(case, kernel, x)
    text = lowered.as_text()
  except Exception as e:  # pylint: disable=broad-exception-caught
    return {"events": list(rec.events), "error": type(e).__name__,
            "message": str(e)[:600]}
  result = rec.results[-1]
  asm = result.module.operation.get_asm(enable_debug_info=True)
  return {
      "events": list(rec.events),
      "seconds_saved": list(rec.saved),
      "normalized": normalize(result),
      "text_sha": hashlib.sha256(text.encode()).hexdigest(),
      "caller_metadata": caller_metadata(lowered),
      "has_file_locations": '.py"' in asm,
      "reads_in_snapshot": sorted({n for n, inside in rec.reads if inside}),
      "reads_outside": sorted({n for n, inside in rec.reads if not inside}),
  }


# -- changes to the process that a case asks for

def apply_bug(name):
  """A bug of the cache, as it would be of its code: not a failure it expects."""
  def boom(*args, **kwargs):
    raise AttributeError(f"a bug of the {name}")
  if name == "encoder":
    cache_encoding.Encoder._e = boom
  elif name == "inputs":
    lowering_cache._gather_inputs = boom
  elif name == "seal":
    cache_seal._hash_file = boom
  else:
    raise ValueError(name)


def wrap(f):
  def wrapper(*args, **kwargs):
    return f(*args, **kwargs)
  return wrapper


def registries_by_name():
  """The registries a lowering dispatches through, by the names a case uses."""
  return {
      "rules": lowering.mosaic_lowering_rules[gpu_core.WGxWG_SEMANTICS],
      "estimators": lowering._resource_estimators,
      "call_primitives": eval_jaxpr_lib.call_primitives,
      "discharge": discharge._discharge_rules,
      "dialect": dialect_lowering._lowerings,
      "warp": dialect_lowering._supported_warp_lowerings,
      "layout": layout_inference._constraint_system_derivation_rules,
  }


def touch_registry(name):
  """Registers a new entry in a registry (as a rule that imports a module does)."""
  registry = registries_by_name()[name]
  if isinstance(registry, (set, frozenset)):
    registry.add(
        jax_core.Primitive("lowering_cache_test_added")
        if name == "call_primitives" else "lowering_cache_test.added")
  elif name in ("rules", "estimators", "discharge"):
    registry[jax_core.Primitive("lowering_cache_test_added")] = next(
        iter(registry.values()))
  else:
    registry["lowering_cache_test.added"] = next(iter(registry.values()))


def apply_tweak(spec):
  name, _, arg = spec.partition(":")
  from jax import lax  # pylint: disable=g-import-not-at-top
  from jax._src.lib import mosaic_gpu_dialect as mgpu_dialect_lib  # pylint: disable=g-import-not-at-top
  from jaxlib.mlir.dialects import arith, vector  # pylint: disable=g-import-not-at-top
  if name == "lane_add_sub":
    # The lane add rule is the lane subtract's.
    table = lowering.mosaic_lowering_rules[gpu_core.LANExWG_SEMANTICS]
    table[lax.add_p] = table[lax.sub_p]
  elif name == "dialect_addf_subf":
    rules = dialect_lowering._lowerings
    rules[arith.AddFOp.OPERATION_NAME] = rules[arith.SubFOp.OPERATION_NAME]
  elif name == "layout_constant_broadcast":
    rules = layout_inference._constraint_system_derivation_rules
    rules[arith.ConstantOp.OPERATION_NAME] = rules[
        vector.BroadcastOp.OPERATION_NAME]
  elif name == "warp_without_async_store":
    dialect_lowering._supported_warp_lowerings.discard(
        mgpu_dialect_lib.AsyncStoreOp.OPERATION_NAME)
  elif name == "foreign_primitive_rule":
    # A rule that is not sealed code, for a primitive of this file's.
    table = lowering.mosaic_lowering_rules[gpu_core.WGxWG_SEMANTICS]
    table[jax_core.Primitive("lowering_cache_test_foreign")] = wrap(
        next(iter(table.values())))
  elif name == "smem_alignment":
    gpu_core.SMEM_ALIGNMENT = int(arg)
  elif name == "smem_alignment_numpy":
    gpu_core.SMEM_ALIGNMENT = np.int64(int(arg))
  elif name == "smem_alignment_object":
    gpu_core.SMEM_ALIGNMENT = object()
  elif name == "smem_alignment_subclass":
    gpu_core.SMEM_ALIGNMENT = _MyInt(int(arg))
  else:
    raise ValueError(spec)


class _MyInt(int):
  """A subclass of int, which a key cannot hold as it holds an int."""


def set_up_touching(case):
  """The stand-in rule registers an entry while the kernel lowers."""
  from jax._src import lowering_cache_test_rules as rules  # pylint: disable=g-import-not-at-top
  rules.HOOKS["registry"] = lambda: touch_registry(case["touch"]["registry"])


def mutate_after_first(case):
  """The change a case makes between its first lowering and its second."""
  mutate = case["mutate"]
  if mutate == "register":
    # A rule that is not sealed code, registered after the registries were first
    # read for a sealed primitive that had none.
    from jax import lax  # pylint: disable=g-import-not-at-top
    table = lowering.mosaic_lowering_rules[gpu_core.WGxWG_SEMANTICS]
    primitive = next(p for _, p in sorted(vars(lax).items(), key=lambda kv: kv[0])
                     if isinstance(p, jax_core.Primitive) and p not in table)
    table[primitive] = wrap(next(iter(table.values())))
    return
  if mutate == "rule":
    table = lowering.mosaic_lowering_rules[gpu_core.WGxWG_SEMANTICS]
  elif mutate == "dialect":
    table = dialect_lowering._lowerings
  elif mutate == "discharge":
    table = discharge._discharge_rules
  else:
    table = layout_inference._constraint_system_derivation_rules
  key = sorted(table, key=str)[0]
  table[key] = wrap(table[key])


def import_late_module(case):
  """A module of the site imported after the process was sealed: one that was
  never listed, one that was listed and edited since, or one that is as listed."""
  how = case["late_module"]
  site = pathlib.Path(case["site"])
  if how == "unlisted":
    name = "jax.experimental.lowering_cache_late"
    (site / "jax/experimental/lowering_cache_late.py").write_text("VALUE = 1\n")
  else:
    name = "jax.experimental.jet"
    if how == "edited":
      path = site / "jax/experimental/jet.py"
      path.write_text(path.read_text() + "\n# edited\n")
  assert name not in sys.modules, name
  importlib.invalidate_caches()
  importlib.import_module(name)


def configure(case, rec):
  """The process, as the case wants it, before anything is lowered."""
  if case.get("smem_bound") is not None:
    mgpu_core._SMEM_SIZE_BOUND = case["smem_bound"]
  for name, value in case.get("config", {}).items():
    jax.config.update(name, value)
  if case.get("nvshmem_bc"):
    # What `is_nvshmem_available` reads: a path in the environment, whether a file
    # is there, and the flag in XLA_FLAGS. (Set once the backend is up: a CPU
    # backend does not know the flag, and stops at it.)
    os.environ["MOSAIC_GPU_NVSHMEM_BC_PATH"] = case["nvshmem_bc"]
    flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = flags + " --xla_gpu_experimental_enable_nvshmem"
  for tweak in case.get("tweaks", ()):
    apply_tweak(tweak)
  if case.get("touch") is not None:
    set_up_touching(case)
  if case.get("mesh") is not None:
    rec.install_topology_spies()
  if case.get("bug") is not None:
    apply_bug(case["bug"])


def run_direct(case, rec, kernel, x):
  """Today's call (the lowering itself, with no cache around it) and the call
  the cache's wrapper makes with the flag unset."""
  wrapper = rec.inner

  def direct(ctx, *args, outer_traceback=None, **kwargs):
    del ctx
    return lowering.lower_unpipelined_jaxpr_to_module(
        *args, outer_traceback=outer_traceback, **kwargs)

  runs = {}
  for mode in ("wrapper", "direct"):
    rec.inner = wrapper if mode == "wrapper" else direct
    runs[mode] = one(case, rec, kernel, x)
  return {"direct": runs, "flag": jax.config.jax_pallas_lowering_cache_dir}


def run(case):
  """What the case reports."""
  if case.get("mode") == "encoder":
    from mosaic_gpu_lowering_cache_support import encoder_cases  # pylint: disable=g-import-not-at-top
    return {"canonicalizer": encoder_cases.run()}
  if case.get("mode") == "audit":
    from mosaic_gpu_lowering_cache_support import state_audit  # pylint: disable=g-import-not-at-top
    return {"audit": state_audit.run()}
  rec = Recorder()
  rec.install()
  # The input is made before the process is changed: a case may change what even
  # the making of an array would use.
  x = make_input(case)
  configure(case, rec)
  kernel = make_kernel(case)
  if case.get("direct"):
    return run_direct(case, rec, kernel, x)
  runs = [one(case, rec, kernel, x) for _ in range(case.get("repeat", 1))]
  between = case.get("mutate") or case.get("after") or case.get("late_module")
  if between:
    if case.get("mutate") is not None:
      mutate_after_first(case)
    for tweak in case.get("after", ()):
      apply_tweak(tweak)
    if case.get("late_module"):
      import_late_module(case)
    runs.append(one(case, rec, kernel, x))
  result = {"runs": runs}
  if case.get("report_seal"):
    seal = cache_seal.seal()
    result["seal"] = {
        "sealed": seal.digest is not None, "reason": seal.reason,
        "seconds": seal.seconds, "files": seal.file_count,
        "bytes": seal.byte_count,
    }
  return result


def main():
  case = json.loads(os.environ["LOWERING_CACHE_CASE"])
  if case.get("site"):
    import jaxlib  # pylint: disable=g-import-not-at-top
    cache_seal._SEAL_ROOTS = [
        case["site"], os.path.dirname(os.path.dirname(jaxlib.__file__))
    ]
  try:
    print("RESULT " + json.dumps(run(case)))
  except Exception as e:  # pylint: disable=broad-exception-caught
    traceback.print_exc()
    print("RESULT " + json.dumps({"crash": f"{type(e).__name__}: {e}"}))


if __name__ == "__main__":
  sys.exit(main())
