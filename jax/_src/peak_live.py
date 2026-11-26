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
"""The buffers live at the peak of each of a compile's allocations.

XLA's buffer assignment names them (``BufferAssignmentProto.peak_buffers``);
the compile states its assignment in its memory analysis
(``serialized_buffer_assignment_proto``: XLA:CPU's, and the CUDA plugin's,
which serves its executables' metadata), which this reads with XLA's own
schema (``hlo_reader``) beside the compiled module the assignment's buffers
are defined in.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import Any

from jax._src import hlo_reader


@dataclasses.dataclass(frozen=True)
class LiveBuffer:
  """A buffer live at the peak of its allocation.

  Attributes:
    bytes: its size in the buffer assignment, as XLA allocates it.
    shape: its dimensions, as compiled (a device's, after partitioning).
    dtype: its element type's XLA name (``f32``, ``bf16``, ...).
    kind: its allocation's: ``parameter`` (an entry parameter's, aliased with
      an output or not), ``output`` or ``temporary``.
    instruction: the name of the instruction defining it.
    op_name: that instruction's ``op_name`` (jax's name stack).
    source: that instruction's source, ``file:line``.
    names: the ``jax.ad_checkpoint.checkpoint_name`` names of the value that
      instruction defines, where it carries them (a fusion its named root's);
      for a while loop's state element an unnamed instruction's buffer holds
      (XLA's buffer aliases: a scan's stacked output, zeros the loop's body
      writes its residuals into), the names of what the body writes into
      that element (``_loop_named``).
    parameter: the entry parameter's number, for a ``parameter`` buffer.
    dims: for each of its dimensions the (argument, axis) pairs it descends
      from, each argument by its flat index among the call's (``args_info``'s
      leaves), through the dimension maps of the ops that made it
      (``provenance``): empty where its size descends from none, None where
      an op on its path states no map (a custom call stating no
      ``output_dims``).
  """
  bytes: int
  shape: tuple[int, ...]
  dtype: str
  kind: str
  instruction: str
  op_name: str
  source: str
  names: tuple[str, ...] = ()
  parameter: int | None = None
  dims: tuple[frozenset[tuple[int, int]] | None, ...] = ()


def live(assignment: bytes, module: bytes,
         argument: Callable[[int], int | None] | None = None
         ) -> tuple[LiveBuffer, ...]:
  """The array buffers live at the peak of each allocation of serialized
  ``BufferAssignmentProto`` ``assignment``, whose buffers serialized
  ``HloModuleProto`` ``module`` defines; a constant's and a thread-local
  allocation's are none."""
  proto = hlo_reader.message_class("xla.BufferAssignmentProto").FromString(
      assignment)
  compiled = hlo_reader.message_class("xla.HloModuleProto").FromString(module)
  instructions = {
      instruction.id: instruction
      for computation in compiled.computations
      for instruction in computation.instructions}
  descended = provenance(compiled, argument)
  computations = {computation.id: computation
                  for computation in compiled.computations}
  # Each logical buffer's other positions, as XLA's must-alias analysis states
  # them: a loop state's buffer's among them, at the while's own output.
  aliases: dict[int, list[Any]] = {}
  for alias in proto.buffer_aliases:
    aliases.setdefault(alias.source_buffer_id, []).append(alias.location)
  primitive_type = hlo_reader.pool().FindEnumTypeByName("xla.PrimitiveType")
  buffers = {buffer.id: buffer for buffer in proto.logical_buffers}
  kinds = {allocation.index: _kind(allocation)
           for allocation in proto.buffer_allocations}
  parameters = {allocation.index: allocation.parameter_number
                for allocation in proto.buffer_allocations
                if allocation.is_entry_computation_parameter}
  found = []
  for peak in proto.peak_buffers:
    kind = kinds[peak.index]
    if kind is None:
      continue
    for buffer_id in peak.logical_buffer_ids:
      buffer = buffers[buffer_id]
      instruction = instructions[buffer.defined_at.instruction_id]
      shape, dims = instruction.shape, descended.get(instruction.id)
      for index in buffer.defined_at.shape_index:
        shape = shape.tuple_shapes[index]
        dims = None if dims is None else dims[index]
      if shape.tuple_shapes:
        continue  # A tuple's index table, no array.
      metadata = instruction.metadata
      named = instruction.frontend_attributes.map.get("checkpoint_name")
      written = None if named else _aliased_named(
          aliases.get(buffer.id, ()), instructions, computations)
      if written is not None:
        named = written.frontend_attributes.map.get("checkpoint_name")
      found.append(LiveBuffer(
          bytes=buffer.size, shape=tuple(shape.dimensions),
          dtype=primitive_type.values_by_number[shape.element_type]
          .name.lower(),
          kind=kind, instruction=instruction.name,
          op_name=metadata.op_name or (
              written.metadata.op_name if written is not None else ""),
          source=f"{metadata.source_file}:{metadata.source_line}",
          names=tuple(named.split(",")) if named else (),
          parameter=parameters.get(peak.index),
          dims=(tuple(None for _ in shape.dimensions) if dims is None
                else tuple(dims))))
  return tuple(found)


def _aliased_named(locations: Any, instructions: dict[int, Any],
                   computations: dict[int, Any]) -> Any | None:
  """The named value a while loop's body writes into a state element a
  buffer is also at (its aliases' ``locations``: a while's output element),
  the first found; None where it is at none or none is named."""
  for location in locations:
    loop = instructions[location.instruction_id]
    if loop.opcode == "while" and len(location.shape_index) == 1:
      written = _loop_named(loop, location.shape_index[0], instructions,
                            computations)
      if written is not None:
        return written
  return None


def _loop_named(loop: Any, element: int, instructions: dict[int, Any],
                computations: dict[int, Any]) -> Any | None:
  """The named value while ``loop``'s body writes into its state's
  ``element``: XLA keeps a loop's state in its initial elements' buffers, so a
  scan's stacked output is the buffer of the zeros its broadcast makes, which
  carries no name, while each step's dynamic-update-slice writes the named
  residual into it (Dew row 13304: a scanned stack's residuals, gibibytes
  each, read unnamed at the step's peak). The body's element where it is
  named; else, for a dynamic-update-slice, fused or not, the update it writes
  (`_written`), named or not, whose op_name says what it is; None where the
  body writes into it no update so read."""
  # XLA's while names its body first, its condition second.
  root = instructions[computations[loop.called_computation_ids[0]].root_id]
  if root.opcode != "tuple" or element >= len(root.operand_ids):
    return None
  written = instructions[root.operand_ids[element]]
  if written.frontend_attributes.map.get("checkpoint_name"):
    return written
  return _written(written, instructions, computations)


def _written(instruction: Any, instructions: dict[int, Any],
             computations: dict[int, Any]) -> Any | None:
  """The update a dynamic-update-slice ``instruction`` writes (its operand 1),
  through a fusion whose root is one: the fused update itself, or, where it
  is a fusion parameter as laid out again (a bitcast, a reshape, a copy, which
  keep its elements), the fusion's operand that parameter stands for; None
  for any other instruction."""
  if instruction.opcode == "dynamic-update-slice":
    return instructions[instruction.operand_ids[1]]
  if instruction.opcode != "fusion":
    return None
  fused = computations[instruction.called_computation_ids[0]]
  root = instructions[fused.root_id]
  if root.opcode != "dynamic-update-slice":
    return None
  update = instructions[root.operand_ids[1]]
  while update.opcode in ("bitcast", "reshape", "copy"):
    update = instructions[update.operand_ids[0]]
  if update.opcode == "parameter":
    return instructions[instruction.operand_ids[update.parameter_number]]
  return update


def _kind(allocation: Any) -> str | None:
  """What allocation ``allocation`` holds: an entry parameter's, an output's or
  temporaries; None for a constant's or a thread-local one."""
  if allocation.is_constant or allocation.is_thread_local:
    return None
  if allocation.is_entry_computation_parameter:
    return "parameter"
  return "output" if allocation.maybe_live_out else "temporary"


Dims = tuple["frozenset[tuple[int, int]] | None", ...]
"""An array's provenance: for each of its dimensions the (argument, axis) pairs
it descends from, empty where its size descends from none (an iota's, a
constant's, a closed-over constant's), None where no op on its path states its
dimensions (a custom call that states none)."""

_SAME_POSITIONS = frozenset((
    "abs", "acos", "acosh", "add", "and", "asin", "asinh", "atan2", "atanh",
    "bitcast-convert", "cbrt", "ceil", "clamp", "compare", "complex",
    "convert", "cosh", "cosine", "count-leading-zeros", "divide", "erf",
    "exp2", "exponential", "exponential-minus-one", "floor", "imag",
    "is-finite", "log", "log-plus-one", "log2", "logistic", "map", "maximum",
    "minimum", "mulhi", "multiply", "negate", "not", "or", "popcnt", "power",
    "real", "reduce-precision", "remainder", "round-nearest-afz",
    "round-nearest-even", "rsqrt", "select", "shift-left",
    "shift-right-arithmetic", "shift-right-logical", "sign", "sine", "sinh",
    "sqrt", "stochastic-convert", "subtract", "tan", "tanh", "xor", "copy",
    "reverse", "cholesky", "triangular-solve", "fft", "slice",
    "dynamic-slice", "pad", "dynamic-update-slice", "concatenate",
    "set-dimension-size", "add-dependency", "domain", "reduce-window",
    "select-and-scatter", "all-reduce", "all-reduce-start", "all-reduce-done",
    "all-gather", "reduce-scatter", "all-to-all", "collective-permute",
    "collective-broadcast", "all-gather-done", "collective-permute-done",
    "copy-done", "async-update",
))
"""Ops each of whose output dimensions descends from its array operands'
dimensions at the same position (every operand of the output's rank): those
computing elementwise, slicing, padding, joining, reversing, or exchanging an
array across devices along a dimension it keeps."""

_DESCENDED_FROM_NONE = frozenset((
    "constant", "iota", "rng-bit-generator", "rng-get-and-update-state",
    "partition-id", "replica-id", "get-dimension-size", "after-all", "infeed",
    "recv", "recv-done", "rng",
))
"""Ops whose outputs' sizes descend from no parameter's axis."""

_PER_OPERAND = frozenset(("sort", "scatter"))
"""Ops whose outputs are their leading operands', one each, as they are."""

_GEMMS = frozenset((
    "__cublas$gemm", "__cublas$lt$matmul", "__cublas$lt$matmul$f8",
))
"""cuBLAS custom calls, whose backend config states their dot's dimension
numbers (``gemm_backend_config.dot_dimension_numbers``)."""


def provenance(module: Any,
               argument: Callable[[int], int | None] | None = None
               ) -> dict[int, Any]:
  """Each instruction's provenance in ``HloModuleProto`` ``module``, by its id:
  an array's ``Dims``, a tuple's a list of its elements'. Read through the
  dimension maps each op states (a dot's and a cuBLAS call's dimension numbers,
  a reduce's, broadcast's and transpose's dimensions, a reshape's and bitcast's
  merges, a gather's and a convolution's dimension numbers), into fusion, call
  and conditional bodies from their callers' operands, and over a while loop's
  body to a fixed point; a custom call states its own in its ``output_dims``
  frontend attribute, else its dimensions are unknown. An instruction a
  computation called from several places reaches holds them all."""
  computations = {computation.id: computation
                  for computation in module.computations}
  found: dict[int, Any] = {}

  def evaluate(computation: Any, parameters: dict[int, Any]) -> Any:
    held: dict[int, Any] = {}
    shapes: dict[int, Any] = {}
    for instruction in computation.instructions:
      operands = [held[operand] for operand in instruction.operand_ids]
      operand_shapes = [shapes[operand] for operand in instruction.operand_ids]
      called = [computations[index]
                for index in instruction.called_computation_ids]
      value = _of(instruction, operands, operand_shapes, parameters, called,
                  evaluate)
      held[instruction.id], shapes[instruction.id] = value, instruction.shape
      found[instruction.id] = (_joined(found[instruction.id], value)
                               if instruction.id in found else value)
    return held[computation.root_id]

  entry = computations[module.entry_computation_id]
  evaluate(entry, {
      instruction.parameter_number: _parameter(
          instruction.parameter_number if argument is None
          else argument(instruction.parameter_number), instruction.shape, ())
      for instruction in entry.instructions
      if instruction.opcode == "parameter"})
  return found


def _of(instruction: Any, operands: list[Any], operand_shapes: list[Any],
        parameters: dict[int, Any], called: list[Any], evaluate: Any) -> Any:
  """``instruction``'s provenance from its operands'."""
  opcode, shape = instruction.opcode, instruction.shape
  if opcode == "parameter":
    return parameters[instruction.parameter_number]
  if opcode in _DESCENDED_FROM_NONE:
    return _filled(shape, frozenset())
  if opcode == "tuple":
    return list(operands)
  if opcode == "opt-barrier":
    return operands[0]
  if opcode == "get-tuple-element":
    return operands[0][instruction.tuple_index]
  if opcode in _SAME_POSITIONS:
    return _same_positions(shape, operands)
  if opcode in _PER_OPERAND:
    return operands[0] if not shape.tuple_shapes else list(
        operands[:len(shape.tuple_shapes)])
  if opcode == "broadcast":
    dims: list[Any] = [frozenset()] * len(shape.dimensions)
    for axis, placed in enumerate(instruction.dimensions):
      dims[placed] = operands[0][axis]
    return tuple(dims)
  if opcode == "transpose":
    return tuple(operands[0][axis] for axis in instruction.dimensions)
  if opcode == "reshape":
    return _merged(operands[0], list(operand_shapes[0].dimensions),
                   list(shape.dimensions))
  if opcode == "bitcast":
    return _bitcast(operands[0], operand_shapes[0], shape)
  if opcode == "dot":
    return _dot(operands[0], operands[1], instruction.dot_dimension_numbers)
  if opcode == "convolution":
    return _convolution(instruction, operands)
  if opcode == "reduce":
    return _reduced(instruction, operands, shape)
  if opcode == "gather":
    return _gathered(instruction, operands)
  if opcode in ("fusion", "call"):
    return evaluate(called[0], dict(enumerate(operands)))
  if opcode == "while":
    return _loop(operands[0], called, evaluate)
  if opcode == "conditional":
    joined = None
    for branch, operand in zip(called, operands[1:]):
      root = evaluate(branch, {0: operand})
      joined = root if joined is None else _joined(joined, root)
    return joined
  if opcode == "async-start":
    # ((its operands), (its wrapped computation's outputs), its context...).
    return [list(operands), evaluate(called[0], dict(enumerate(operands))),
            *(_filled(element, frozenset())
              for element in shape.tuple_shapes[2:])]
  if opcode in ("all-gather-start", "collective-permute-start", "copy-start"):
    # (its operand, its output, its context...), the output its operand's.
    head = operands[0] if len(operands) == 1 else list(operands)
    return [head, head, *(_filled(element, frozenset())
                          for element in shape.tuple_shapes[2:])]
  if opcode == "async-done":
    return operands[0][1]
  if opcode == "custom-call":
    return _custom(instruction, operands, shape)
  return _filled(shape, None)


def _parameter(number: int | None, shape: Any, index: tuple[int, ...]) -> Any:
  """An entry parameter's provenance, numbered as the call's argument it is
  (``number``): each dimension its own axis; none for a parameter that is no
  argument (a closed-over constant, a token), whose sizes descend from none;
  an element of a tuple parameter by its index, which the pairs do not name,
  unknown."""
  if shape.tuple_shapes:
    return [_parameter(number, element, (*index, position))
            for position, element in enumerate(shape.tuple_shapes)]
  if number is None:
    return tuple(frozenset() for _ in shape.dimensions)
  if index:
    return tuple(None for _ in shape.dimensions)
  return tuple(frozenset({(number, axis)})
               for axis in range(len(shape.dimensions)))


def _filled(shape: Any, dimension: Any) -> Any:
  """``shape``'s provenance with every dimension ``dimension``."""
  if shape.tuple_shapes:
    return [_filled(element, dimension) for element in shape.tuple_shapes]
  return tuple(dimension for _ in shape.dimensions)


def _union(left: Any, right: Any) -> Any:
  """Two dimensions' provenance together: unknown where either is."""
  return None if left is None or right is None else left | right


def _joined(left: Any, right: Any) -> Any:
  """Two provenances of one shape together, dimension by dimension."""
  if isinstance(left, list) and isinstance(right, list):
    return [_joined(a, b) for a, b in zip(left, right)]
  if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(
      right):
    return tuple(_union(a, b) for a, b in zip(left, right))
  raise ValueError(f"provenances of two shapes joined: {left!r}, {right!r}")


def _same_positions(shape: Any, operands: list[Any]) -> Any:
  """Each output dimension its operands' at its position: a tuple output's
  elements each their tuple operands' element, or the array operand at their
  index (a variadic collective's, reduce-window's)."""
  if shape.tuple_shapes:
    tuples = [operand for operand in operands if isinstance(operand, list)]
    return [_same_positions(element, [operand[index] for operand in tuples]
                            if tuples else [operands[index]])
            for index, element in enumerate(shape.tuple_shapes)]
  rank = len(shape.dimensions)
  dims: list[Any] = [frozenset()] * rank
  for operand in operands:
    if isinstance(operand, tuple) and len(operand) == rank:
      dims = [_union(held, given) for held, given in zip(dims, operand)]
  return tuple(dims)


def _merged(operand: Dims, sizes: list[int], merged: list[int]) -> Dims:
  """A reshape's provenance: each output dimension descends from each operand
  dimension whose span of the row-major index it overlaps (a dimension spans
  the strides from the product of the sizes after it to that times its own),
  so a merged dimension from all it merges and a split one's parts each from
  the one it splits; a dimension of size one spans none."""
  def spans(shape: list[int]) -> list[tuple[int, int]]:
    found, stride = [], 1
    for size in reversed(shape):
      found.append((stride, stride * size))
      stride *= size
    return list(reversed(found))

  given = spans(sizes)
  dims = []
  for low, high in spans(merged):
    descended: Any = frozenset()
    for axis, (start, end) in enumerate(given):
      if max(low, start) < min(high, end):
        descended = _union(descended, operand[axis])
    dims.append(descended)
  return tuple(dims)


def _bitcast(operand: Dims, operand_shape: Any, shape: Any) -> Dims:
  """A bitcast's provenance: a reshape of the dimensions as laid out, major to
  minor (its layout's), so a bitcast that transposes is read as one."""
  if list(operand_shape.dimensions) == list(shape.dimensions) and list(
      operand_shape.layout.minor_to_major) == list(shape.layout.minor_to_major):
    return operand
  source = _major_to_minor(operand_shape)
  target = _major_to_minor(shape)
  laid = _merged(tuple(operand[axis] for axis in source),
                 [operand_shape.dimensions[axis] for axis in source],
                 [shape.dimensions[axis] for axis in target])
  dims: list[Any] = [frozenset()] * len(shape.dimensions)
  for position, axis in enumerate(target):
    dims[axis] = laid[position]
  return tuple(dims)


def _major_to_minor(shape: Any) -> list[int]:
  order = list(shape.layout.minor_to_major)
  return list(reversed(order)) if order else list(range(len(shape.dimensions)))


def _dot(lhs: Dims, rhs: Dims, numbers: Any) -> Dims:
  """A dot's provenance: its batch dimensions, then the left operand's free
  ones, then the right's, each from those it is made of."""
  lhs_batch, rhs_batch = list(numbers.lhs_batch_dimensions), list(
      numbers.rhs_batch_dimensions)
  lhs_kept = set(lhs_batch) | set(numbers.lhs_contracting_dimensions)
  rhs_kept = set(rhs_batch) | set(numbers.rhs_contracting_dimensions)
  return (*(_union(lhs[a], rhs[b]) for a, b in zip(lhs_batch, rhs_batch)),
          *(lhs[axis] for axis in range(len(lhs)) if axis not in lhs_kept),
          *(rhs[axis] for axis in range(len(rhs)) if axis not in rhs_kept))


def _convolution(instruction: Any, operands: list[Any]) -> Dims:
  """A convolution's provenance: its batch dimension the input's, its feature
  dimension the kernel's output feature (with the input's feature or batch
  dimension where it groups them), each spatial dimension the input's and the
  kernel's at its place."""
  numbers, (lhs, rhs) = instruction.convolution_dimension_numbers, operands[:2]
  dims: list[Any] = [frozenset()] * (2 + len(numbers.output_spatial_dimensions))
  dims[numbers.output_batch_dimension] = lhs[numbers.input_batch_dimension]
  feature = rhs[numbers.kernel_output_feature_dimension]
  if instruction.feature_group_count > 1:
    feature = _union(feature, lhs[numbers.input_feature_dimension])
  if instruction.batch_group_count > 1:
    feature = _union(feature, lhs[numbers.input_batch_dimension])
  dims[numbers.output_feature_dimension] = feature
  for output, given, kernel in zip(numbers.output_spatial_dimensions,
                                   numbers.input_spatial_dimensions,
                                   numbers.kernel_spatial_dimensions):
    dims[output] = _union(lhs[given], rhs[kernel])
  return tuple(dims)


def _reduced(instruction: Any, operands: list[Any], shape: Any) -> Any:
  """A reduce's provenance: each output its input's dimensions but those it
  reduces, in order."""
  inputs = operands[:len(operands) // 2]
  removed = set(instruction.dimensions)
  outputs = [tuple(dimension for axis, dimension in enumerate(operand)
                   if axis not in removed) for operand in inputs]
  return outputs if shape.tuple_shapes else outputs[0]


def _gathered(instruction: Any, operands: list[Any]) -> Dims:
  """A gather's provenance: its offset dimensions the operand's it keeps, in
  order; its batch dimensions the start indices', in order, with an operand
  batching dimension's beside its indices' own."""
  numbers, (operand, indices) = instruction.gather_dimension_numbers, operands[:2]
  offsets = list(numbers.offset_dims)
  dropped = set(numbers.collapsed_slice_dims) | set(numbers.operand_batching_dims)
  kept = [axis for axis in range(len(operand)) if axis not in dropped]
  batched = dict(zip(numbers.start_indices_batching_dims,
                     numbers.operand_batching_dims))
  indexed = [axis for axis in range(len(indices))
             if axis != numbers.index_vector_dim]
  rank = len(offsets) + len(indexed)
  dims: list[Any] = [frozenset()] * rank
  batch = iter(indexed)
  offset = iter(kept)
  for axis in range(rank):
    if axis in offsets:
      dims[axis] = operand[next(offset)]
    else:
      index = next(batch)
      dims[axis] = indices[index]
      if index in batched:
        dims[axis] = _union(dims[axis], operand[batched[index]])
  return tuple(dims)


def _loop(initial: Any, called: list[Any], evaluate: Any) -> Any:
  """A while loop's provenance: its state's, its body's output joined into it
  until the body adds nothing (sets only grow, so it ends)."""
  body = called[0]  # XLA's while names its body first, its condition second.
  state = initial
  while True:
    joined = _joined(state, evaluate(body, {0: state}))
    if joined == state:
      return state
    state = joined


def _custom(instruction: Any, operands: list[Any], shape: Any) -> Any:
  """A custom call's provenance: as its ``output_dims`` frontend attribute
  states it (one list an output, one entry a dimension, each a list of
  [operand, axis] pairs, empty for a size descending from none, null for one
  unknown); a cuBLAS call's from its dot's dimension numbers, its workspace's
  from none; else unknown."""
  stated = instruction.frontend_attributes.map.get("output_dims")
  if stated:
    elements = list(shape.tuple_shapes) if shape.tuple_shapes else [shape]
    aliased = {tuple(alias.output_shape_index): alias.operand_index
               for alias in instruction.output_operand_aliasing}
    given = json.loads(stated)
    outputs = []
    for index, element in enumerate(elements):
      if (index,) in aliased:
        # An operand updated in place (a ref a kernel adds into): its own.
        outputs.append(operands[aliased[(index,)]])
      elif index < len(given) and given[index] is not None:
        outputs.append(tuple(
            None if entry is None else _from_operands(entry, operands)
            for entry in given[index]))
      else:
        # Stated unknown (null), or past the outputs stated (a kernel's
        # scratch): unknown.
        outputs.append(_filled(element, None))
    return outputs if shape.tuple_shapes else outputs[0]
  if instruction.custom_call_target in _GEMMS:
    config = json.loads(instruction.backend_config or "{}")
    numbers = config.get("gemm_backend_config", {}).get("dot_dimension_numbers")
    if numbers is not None:
      result = _dot(operands[0], operands[1], _Numbers(numbers))
      if shape.tuple_shapes:
        return [result, *(_filled(element, frozenset())
                          for element in shape.tuple_shapes[1:])]
      return result
  return _filled(shape, None)


def _from_operands(entry: list[list[int]], operands: list[Any]) -> Any:
  descended: Any = frozenset()
  for operand, axis in entry:
    descended = _union(descended, operands[operand][axis])
  return descended


class _Numbers:
  """A dot's dimension numbers as a cuBLAS call's backend config states them
  (JSON: lists of strings), read as ``DotDimensionNumbers`` is."""

  def __init__(self, numbers: dict[str, list[str]]):
    for field in ("lhs_contracting_dimensions", "rhs_contracting_dimensions",
                  "lhs_batch_dimensions", "rhs_batch_dimensions"):
      setattr(self, field, [int(axis) for axis in numbers.get(field, [])])
