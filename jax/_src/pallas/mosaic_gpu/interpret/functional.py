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
"""A functional interpreter for Pallas Mosaic GPU kernels of one thread a block.

The GPU interpreter (`interpret_pallas_call`) runs each thread of a block on its
own CPU thread and keeps every buffer in a simulated memory on the host, so
every load and store is an `io_callback` (about 170 us each on a CPU host; a
kernel of a few thousand accesses a call spends nearly all of its time in
them). A kernel whose blocks run one thread (warpgroup), in clusters of one
block, on one device needs none of that: its accesses happen in program order.
This interpreter runs such a kernel as one JAX program: every buffer is an
array carried through it, at its logical shape (as the GPU interpreter stores
it), a load or store indexes it, the grid is a
`fori_loop`, and asynchronous copies complete as they are issued, so barriers,
commits and waits do nothing. A kernel whose warps run apart (a `WarpMesh`
`mpmd_map`) runs on the simulated memory, which runs them at once. Buffers start as the GPU
interpreter's do (`uninitialized_memory`), and an out-of-bounds access is caught
as it is (`out_of_bounds_reads`): flagged through the program and raised once,
at its end, with no callback per access.

It is used where `InterpretGPUParams.functional` allows (`eligible`): never
with `detect_races`, which needs the host memory's vector clocks.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import Any

import jax
from jax import lax
from jax._src import callback
from jax._src import core as jax_core
from jax._src import source_info_util
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic_gpu import core as mosaic_gpu_core
from jax._src.pallas.mosaic_gpu import primitives as gpu_primitives
from jax._src.pallas.mosaic_gpu.interpret import jaxpr_interpret
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax._src.state import types as state_types
from jax._src.state import utils as state_utils
from jax._src.util import safe_zip
from jax._src.util import split_list
from jax._src.pallas import mpmd
import jax.numpy as jnp
import numpy as np


_NO_OPS = (
    gpu_primitives.barrier_arrive_p,
    gpu_primitives.barrier_wait_p,
    gpu_primitives.barrier_arrive_and_wait_p,
    gpu_primitives.commit_smem_p,
    gpu_primitives.wait_smem_to_gmem_p,
    gpu_primitives.commit_group_p,
    gpu_primitives.set_max_registers_p,
    gpu_primitives.async_prefetch_p,
)
"""Ordering only: in one thread, in program order, with copies that complete as
they are issued, each of these has nothing left to order."""

_HANDLED = (
    lax.axis_index_p,
    primitives.program_id_p,
    state_primitives.get_p,
    gpu_primitives.load_p,
    state_primitives.swap_p,
    gpu_primitives.store_p,
    primitives.run_scoped_p,
    lax.cond_p,
    lax.scan_p,
    lax.while_p,
    gpu_primitives.copy_gmem_to_smem_p,
    gpu_primitives.copy_smem_to_gmem_p,
    mosaic_gpu_core.layout_cast_p,
    *_NO_OPS,
)

_REFUSED = frozenset(
    p
    for p in vars(gpu_primitives).values()
    if isinstance(p, jax_core.Primitive)
) - frozenset(_HANDLED)
"""Every other Mosaic GPU primitive (MMAs, TMEM, cluster cancellation, inline
MGPU, ...): a kernel holding one runs on the GPU interpreter."""


def _jaxprs(params: dict[str, Any]):
  for value in params.values():
    for inner in value if isinstance(value, (tuple, list)) else (value,):
      if isinstance(inner, jax_core.Jaxpr):
        yield inner


def unsupported(jaxpr: jax_core.Jaxpr) -> str | None:
  """The first primitive of `jaxpr`, at any depth, this interpreter does not
  run, by name; None where it runs them all."""
  for eqn in jaxpr.eqns:
    if eqn.primitive in _REFUSED or eqn.primitive in (
        primitives.load_p, primitives.swap_p, mpmd.mpmd_map_p):
      # mpmd_map: a warpgroup's warps run at once, share SMEM and synchronize.
      return eqn.primitive.name
    if eqn.primitive is gpu_primitives.copy_gmem_to_smem_p and any(
        isinstance(t, mosaic_gpu_core.Im2ColTransform)
        for t in jax.tree.leaves(eqn.params["src_transforms_treedef"])):
      return "copy_gmem_to_smem (im2col)"
    for inner in _jaxprs(eqn.params):
      if (found := unsupported(inner)) is not None:
        return found
  return None


def eligible(
    jaxpr: jax_core.Jaxpr,
    grid_mapping: pallas_core.GridMapping,
    num_threads_per_block: int,
    cluster_dims: tuple[int, ...],
    num_devices: int,
    interpret_params: InterpretGPUParams,
) -> bool:
  """Whether this interpreter runs the kernel: one thread a block, clusters of
  one block, one device, its inputs and outputs in GMEM, no race detection, and
  every primitive one it runs (`unsupported`). `functional=True` makes a kernel
  it cannot run an error, `False` keeps the GPU interpreter."""
  if interpret_params.functional is False or interpret_params.detect_races:
    return False
  why = None
  if num_threads_per_block != 1:
    why = f"{num_threads_per_block} threads a block"
  elif math.prod(cluster_dims) != 1:
    why = f"clusters of {math.prod(cluster_dims)} blocks"
  elif num_devices != 1:
    why = f"{num_devices} devices"
  elif not all(
      _is_gmem(var.aval)
      for var in jaxpr.invars[: grid_mapping.num_inputs + grid_mapping.num_outputs]
  ):
    why = "an input or output outside GMEM"
  elif (found := unsupported(jaxpr)) is not None:
    why = f"its {found}"
  if why is not None and interpret_params.functional:
    raise NotImplementedError(
        f"The functional GPU interpreter does not run this kernel: {why}.")
  return why is None


def _is_gmem(aval) -> bool:
  space = getattr(aval, "memory_space", None)
  return space in (None, mosaic_gpu_core.MemorySpace.GMEM, pallas_core.MemorySpace.ANY)


@dataclasses.dataclass(frozen=True)
class _Ref:
  """A buffer: its slot in the carried state, or a `RefUnion`'s members' slots."""
  slots: tuple[int, ...]
  union: tuple[Any, ...] | None = None  # the members' `ExtractAliasedRef`s


class _Barrier:
  """A barrier: nothing to keep, in one thread."""


def _fill(shape, dtype, params: InterpretGPUParams, space=None):
  how = "zero" if space == mosaic_gpu_core.MemorySpace.REGS else params.uninitialized_memory
  return interpret_utils.get_uninitialized_array(shape, dtype, how)


def _in_bounds(shape, indexer: indexing.NDIndexer):
  """Whether every element `indexer` names lies within `shape`."""
  ok = jnp.bool_(True)
  for dim, index in safe_zip(shape, indexer.indices):
    if isinstance(index, indexing.Slice):
      if index.size:
        ok &= (index.start >= 0) & (index.start + (index.size - 1) * index.stride < dim)
    else:
      index = jnp.asarray(index)
      ok &= jnp.all((index >= 0) & (index < dim))
  return ok


def _positions(indexer: indexing.NDIndexer):
  """Each indexed dimension's positions, broadcastable as the read's shape (a
  Slice's along its own axis), or None where an array index makes it not a
  rectangle."""
  sliced = [i for i in indexer.indices if isinstance(i, indexing.Slice)]
  if any(not isinstance(i, indexing.Slice) and jnp.ndim(i) for i in indexer.indices):
    return None
  axis, found = 0, []
  for index in indexer.indices:
    if isinstance(index, indexing.Slice):
      position = index.start + index.stride * lax.iota(jnp.int32, index.size)
      found.append(position.reshape((1,) * axis + (-1,) + (1,) * (len(sliced) - axis - 1)))
      axis += 1
    else:
      found.append(jnp.asarray(index))
  return found


def _gathered(value, indexer, fill):
  """`value` at `indexer`, `fill` where it reaches past `value`."""
  positions = _positions(indexer)
  if positions is None:
    raise NotImplementedError(
        "The functional GPU interpreter reads past a buffer only through slices"
        " and scalar indices.")
  # A position before the buffer (a window's negative start) lies past it too,
  # not wrapped round to its end.
  return value.at[tuple(positions)].get(
      mode="fill", fill_value=fill, wrap_negative_indices=False)


def _scattered(value, indexer, val):
  """`value` with `val` written at `indexer`, the part past `value` dropped."""
  positions = _positions(indexer)
  if positions is None:
    raise NotImplementedError(
        "The functional GPU interpreter writes past a buffer only through slices"
        " and scalar indices.")
  return value.at[tuple(positions)].set(
      val, mode="drop", wrap_negative_indices=False)


def _forward(x, transform):
  """`x` seen through a transform other than indexing."""
  match transform:
    case state_types.TransposeTransform():
      return jnp.transpose(x, transform.permutation)
    case state_types.BitcastTransform():
      return state_utils.bitcast(x, transform.dtype)
    case state_types.ReshapeTransform():
      return lax.reshape(x, transform.shape)
  return transform.transform_array(x)


def _backward(seen, transform, x):
  """What `x` holds once `seen` (it through `transform`) is written."""
  match transform:
    case state_types.TransposeTransform():
      return jnp.transpose(seen, np.argsort(transform.permutation))
    case state_types.BitcastTransform():
      return state_utils.bitcast(seen, x.dtype)
    case state_types.ReshapeTransform():
      return lax.reshape(seen, x.shape)
  return transform.undo(x).transform_array(seen)


class _Interpreter:
  """One kernel's interpretation as one program: `state` holds every buffer's
  value; `oob` whether any access went out of bounds."""

  def __init__(self, params: InterpretGPUParams, axes: dict[Any, Any]):
    self.params = params
    self.axes = axes
    self.state: list[Any] = []
    self.oob = jnp.bool_(False)

  # Buffers.

  def allocate(self, value) -> int:
    self.state.append(value)
    return len(self.state) - 1

  def _slot(self, ref: _Ref, transforms):
    if ref.union is None:
      return ref.slots[0], transforms
    extract, *rest = transforms
    return ref.slots[ref.union.index(extract)], tuple(rest)

  def read(self, ref: _Ref, transforms, overhang=None):
    """The value at `transforms` of `ref`. Out of bounds: `overhang` where given
    (a TMA load's zeros), else as `out_of_bounds_reads` says."""
    slot, transforms = self._slot(ref, transforms)
    value = self.state[slot]
    for position, transform in enumerate(transforms):
      if not isinstance(transform, indexing.NDIndexer):
        value = _forward(value, transform)
        continue
      last = position == len(transforms) - 1
      if overhang is not None and last:
        value = _gathered(value, transform, overhang)
      elif self.params.out_of_bounds_reads == "raise" or not last:
        self.oob |= ~_in_bounds(value.shape, transform)
        value = state_discharge.transform_array(value, [transform])
      else:
        value = _gathered(value, transform, interpret_utils.get_uninitialized_value(
            value.dtype, self.params.uninitialized_memory))
    return value

  def write(self, ref: _Ref, transforms, val, drop: bool = False):
    """`val` written at `transforms` of `ref`, returning what it replaced. Out of
    bounds: dropped where `drop` (a TMA store's overhang past GMEM), else an
    error raised at the program's end."""
    slot, transforms = self._slot(ref, transforms)
    transforms = tuple(transforms)
    if drop and transforms and isinstance(transforms[-1], indexing.NDIndexer):
      *outer, last = transforms
      inner = state_discharge.transform_array(self.state[slot], outer) if outer else self.state[slot]
      updated = _scattered(inner, last, val)
      if outer:
        _, self.state[slot] = state_discharge.transform_swap_array(self.state[slot], outer, updated)
      else:
        self.state[slot] = updated
      return None
    old, self.state[slot] = self._swapped(self.state[slot], transforms, val)
    return old

  def _swapped(self, x, transforms, val):
    """`val` written at `transforms` of `x`: the value it replaced, and `x` after."""
    if not transforms:
      return x, val
    transform, rest = transforms[0], transforms[1:]
    if isinstance(transform, indexing.NDIndexer):
      self.oob |= ~_in_bounds(x.shape, transform)
      inner = state_discharge.transform_array(x, [transform])
      old, inner = self._swapped(inner, rest, val)
      _, x = state_discharge.transform_swap_array(x, [transform], inner)
      return old, x
    old, inner = self._swapped(_forward(x, transform), rest, val)
    return old, _backward(inner, transform, x)

  # Evaluation.

  def run(self, jaxpr: jax_core.Jaxpr, *args):
    env: dict[Any, Any] = {}

    def read(var):
      return var.val if isinstance(var, jax_core.Literal) else env[var]

    for var, value in safe_zip(jaxpr.constvars + jaxpr.invars, args):
      env[var] = value
    for eqn in jaxpr.eqns:
      with source_info_util.user_context(
          eqn.source_info.traceback, name_stack=eqn.source_info.name_stack):
        out = self.eqn(eqn, [read(v) for v in eqn.invars])
      out = out if eqn.primitive.multiple_results else [out]
      for var, value in safe_zip(eqn.outvars, out):
        env[var] = value
    return [read(v) for v in jaxpr.outvars]

  def eqn(self, eqn, invals):
    p, params = eqn.primitive, eqn.params
    if p is lax.axis_index_p:
      if params["axis_name"] not in self.axes:
        raise ValueError(f"Unable to determine axis index for axis name {params['axis_name']}")
      return self.axes[params["axis_name"]]
    if p is primitives.program_id_p:
      return jnp.int32(0)
    if p in (state_primitives.get_p, gpu_primitives.load_p):
      return self.read(invals[0], jax.tree.unflatten(params["tree"], invals[1:]))
    if p in (state_primitives.swap_p, gpu_primitives.store_p):
      old = self.write(invals[0], jax.tree.unflatten(params["tree"], invals[2:]), invals[1])
      return [] if p is gpu_primitives.store_p else old
    if p is mosaic_gpu_core.layout_cast_p:
      return invals[0]
    if p in _NO_OPS:
      return [] if p.multiple_results else None
    if p is primitives.run_scoped_p:
      return self.run_scoped(eqn, invals)
    if p is lax.cond_p:
      return self.cond(eqn, invals)
    if p is lax.scan_p:
      return self.scan(eqn, invals)
    if p is lax.while_p:
      return self.while_(eqn, invals)
    if p is gpu_primitives.copy_gmem_to_smem_p:
      src, dst, *flat = invals
      if params["has_barrier"]:
        flat = flat[1:]  # Its barrier: one program runs the copy at once.
      predicate = flat.pop() if params["has_user_predicate"] else None
      src_t, dst_t, _ = jax.tree.unflatten(
          jax.tree_util.treedef_tuple((
              params["src_transforms_treedef"], params["dst_transforms_treedef"],
              params["barrier_transforms_treedef"])), flat)
      # A TMA load reads zeros past its GMEM source; its SMEM window must fit.
      # A fill is a constant (`.at[].get(mode="fill")`), never a traced zero.
      zero = np.zeros((), self.state[src.slots[0]].dtype)[()]

      def copied(state, oob):
        inner = _Interpreter(self.params, self.axes)
        inner.state, inner.oob = list(state), oob
        inner.write(dst, dst_t, inner.read(src, src_t, overhang=zero))
        return tuple(inner.state), inner.oob

      if predicate is None:
        state, oob = copied(tuple(self.state), self.oob)
      else:
        # A copy its predicate disables touches no memory, in bounds or not.
        state, oob = lax.cond(predicate, copied, lambda state, oob: (state, oob),
                              tuple(self.state), self.oob)
      self.state, self.oob = list(state), oob
      return []
    if p is gpu_primitives.copy_smem_to_gmem_p:
      src, dst, *flat = invals
      predicate = flat.pop() if params["has_user_predicate"] else None
      src_t, dst_t = jax.tree.unflatten(
          jax.tree_util.treedef_tuple((
              params["src_transforms_treedef"], params["dst_transforms_treedef"])), flat)
      value = self.read(src, src_t)
      if params.get("reduction_op") is not None:
        raise NotImplementedError("copy_smem_to_gmem with a reduction")
      if predicate is not None:
        value = jnp.where(predicate, value, self.read(dst, dst_t, overhang=np.zeros((), value.dtype)[()]))
      # A TMA store drops what lies past its GMEM destination.
      self.write(dst, dst_t, value, drop=True)
      return []
    return p.bind(*invals, **p.get_bind_params(params))

  def run_scoped(self, eqn, invals):
    jaxpr = eqn.params["jaxpr"]
    depth = len(self.state)
    allocs = []
    # Each at its logical shape, as the GPU interpreter allocates a scoped
    # buffer.
    for var in jaxpr.invars:
      aval = var.aval
      if isinstance(aval, mosaic_gpu_core.AbstractRefUnion):
        # A buffer a member, each at its logical shape, as the simulated memory
        # allocates a union (`jaxpr_interpret.allocate_ref_union`).
        members = jaxpr_interpret.union_members(aval)
        slots = []
        for member in members:
          inner = jaxpr_interpret._member_inner_aval(member)
          slots.append(self.allocate(_fill(inner.shape, inner.dtype, self.params, aval.memory_space)))
        allocs.append(_Ref(tuple(slots), union=tuple(m.transforms[0] for m in members)))
        continue
      inner = aval.inner_aval
      if isinstance(inner.dtype, (mosaic_gpu_core.BarrierType, mosaic_gpu_core.ClusterBarrierType)):
        allocs.append(_Barrier())
        continue
      allocs.append(_Ref((self.allocate(
          _fill(inner.shape, inner.dtype, self.params, aval.memory_space)),)))
    out = self.run(jaxpr, *invals, *allocs)
    del self.state[depth:]
    return out

  # Control flow, its body a function of the state it reads and writes.

  def _arrays(self, values):
    return [v for v in values if not isinstance(v, (_Ref, _Barrier))]

  def _rejoined(self, values, arrays):
    arrays = iter(arrays)
    return [v if isinstance(v, (_Ref, _Barrier)) else next(arrays) for v in values]

  def _body(self, jaxpr, operands):
    """`jaxpr` as a function of (state, oob, the operands' arrays) to (state,
    oob, its outputs), the operands' buffers fixed."""
    def body(state, oob, arrays):
      inner = _Interpreter(self.params, self.axes)
      inner.state, inner.oob = list(state), oob
      out = inner.run(jaxpr, *self._rejoined(operands, arrays))
      return tuple(inner.state), inner.oob, out
    return body

  def cond(self, eqn, invals):
    index, *operands = invals
    branches = [
        functools.partial(self._body(branch.jaxpr, operands)) for branch in eqn.params["branches"]]
    state, oob, out = lax.switch(
        index, branches, tuple(self.state), self.oob, self._arrays(operands))
    self.state, self.oob = list(state), oob
    return out

  def scan(self, eqn, invals):
    params = eqn.params
    consts, carry, xs = (list(g) for g in params["ft_in"].update(invals).unpack())

    def step(loop, x):
      state, oob, c = loop
      operands = [*consts, *c, *x]
      state, oob, out = self._body(params["jaxpr"].jaxpr, operands)(state, oob, self._arrays(operands))
      new_c, y = (list(g) for g in params["ft_out"].update(out).unpack())
      return (state, oob, new_c), y

    (state, oob, carry), ys = lax.scan(
        step, (tuple(self.state), self.oob, carry), xs, length=params.get("length"),
        reverse=params.get("reverse", False))
    self.state, self.oob = list(state), oob
    return [*carry, *ys]

  def while_(self, eqn, invals):
    params = eqn.params
    cond_consts, body_consts, init = split_list(
        invals, [params["cond_nconsts"], params["body_nconsts"]])
    cond_jaxpr, body_jaxpr = params["cond_jaxpr"].jaxpr, params["body_jaxpr"].jaxpr

    def tested(state, oob, vals):
      # The condition runs as the kernel's code does, its writes and its
      # out-of-bounds reads kept with the loop's state.
      operands = [*cond_consts, *vals]
      state, oob, (going,) = self._body(cond_jaxpr, operands)(
          state, oob, self._arrays(operands))
      return state, oob, going

    def step(loop):
      state, oob, vals, _ = loop
      operands = [*body_consts, *vals]
      state, oob, out = self._body(body_jaxpr, operands)(state, oob, self._arrays(operands))
      state, oob, going = tested(state, oob, out)
      return state, oob, out, going

    state, oob, going = tested(tuple(self.state), self.oob, init)
    state, oob, out, _ = lax.while_loop(
        lambda loop: loop[3], step, (state, oob, init, going))
    self.state, self.oob = list(state), oob
    return out


def interpret(
    *args,
    jaxpr: jax_core.Jaxpr,
    grid_mapping: pallas_core.GridMapping,
    mesh: mosaic_gpu_core.Mesh | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    interpret_params: InterpretGPUParams,
    grid_dims: tuple[int, ...],
    device_axes: dict[Any, int],
) -> Sequence[jax.Array]:
  """The kernel `jaxpr` over `args`, interpreted as one program (`eligible`)."""
  num_inputs, num_outputs = grid_mapping.num_inputs, grid_mapping.num_outputs
  inputs = list(args[len(args) - num_inputs:])
  interpreter = _Interpreter(interpret_params, {})
  input_refs = [_Ref((interpreter.allocate(x),)) for x in inputs]
  aliases = {out: inp for inp, out in input_output_aliases}
  output_refs = []
  for index, bm in enumerate(grid_mapping.block_mappings_output):
    if index in aliases:
      output_refs.append(input_refs[aliases[index]])
    else:
      output_refs.append(_Ref((interpreter.allocate(_fill(
          bm.array_aval.shape, bm.array_aval.dtype, interpret_params)),)))
  # Each scratch buffer at its logical shape, as BlockSpec and run_scoped
  # allocate it.
  scratch_refs = []
  for var in jaxpr.invars[num_inputs + num_outputs:]:
    inner = var.aval.inner_aval
    scratch_refs.append(_Ref((interpreter.allocate(
        _fill(inner.shape, inner.dtype, interpret_params, var.aval.memory_space)),)))
  kernel_refs = [*input_refs, *output_refs, *scratch_refs]

  # The axes an `axis_index` names, as the GPU interpreter reads them: the
  # grid's by the block's coordinates, the one block of a cluster's and the one
  # thread's as 0, and the device's.
  fixed = {name: jnp.int32(index) for name, index in device_axes.items()}
  if mesh is not None:
    fixed |= {name: jnp.int32(0) for name in mesh.cluster_names or ()}
    if mesh.thread_name is not None:
      fixed[mesh.thread_name] = jnp.int32(0)
  grid_names = tuple(mesh.grid_names) if mesh is not None else ()

  def block(index, loop):
    state, oob = loop
    coords = interpret_utils.get_indices(grid_dims, index)
    axes = {**fixed, **{name: jnp.int32(c) for name, c in zip(grid_names, coords)}}
    inner = _Interpreter(interpret_params, axes)
    inner.state, inner.oob = list(state), oob
    inner.run(jaxpr, *kernel_refs)
    return tuple(inner.state), inner.oob

  state, oob = lax.fori_loop(
      0, math.prod(grid_dims), block, (tuple(interpreter.state), interpreter.oob))
  if interpret_params.out_of_bounds_reads == "raise":
    def raise_if(out_of_bounds):
      if out_of_bounds:
        raise IndexError("Out-of-bounds read or store in a Pallas GPU kernel"
                         " (functional GPU interpreter).")
      return np.int32(0)
    callback.io_callback(raise_if, jax.ShapeDtypeStruct((), jnp.int32), oob, ordered=True)
  return [state[ref.slots[0]] for ref in output_refs]
