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
      instruction defines, where it carries them (a fusion its named root's).
    parameter: the entry parameter's number, for a ``parameter`` buffer.
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


def live(assignment: bytes, module: bytes) -> tuple[LiveBuffer, ...]:
  """The array buffers live at the peak of each allocation of serialized
  ``BufferAssignmentProto`` ``assignment``, whose buffers serialized
  ``HloModuleProto`` ``module`` defines; a constant's and a thread-local
  allocation's are none."""
  proto = hlo_reader.message_class("xla.BufferAssignmentProto").FromString(
      assignment)
  instructions = {
      instruction.id: instruction
      for computation in hlo_reader.message_class("xla.HloModuleProto")
      .FromString(module).computations
      for instruction in computation.instructions}
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
      shape = instruction.shape
      for index in buffer.defined_at.shape_index:
        shape = shape.tuple_shapes[index]
      if shape.tuple_shapes:
        continue  # A tuple's index table, no array.
      metadata = instruction.metadata
      named = instruction.frontend_attributes.map.get("checkpoint_name")
      found.append(LiveBuffer(
          bytes=buffer.size, shape=tuple(shape.dimensions),
          dtype=primitive_type.values_by_number[shape.element_type]
          .name.lower(),
          kind=kind, instruction=instruction.name,
          op_name=metadata.op_name,
          source=f"{metadata.source_file}:{metadata.source_line}",
          names=tuple(named.split(",")) if named else (),
          parameter=parameters.get(peak.index)))
  return tuple(found)


def _kind(allocation: Any) -> str | None:
  """What allocation ``allocation`` holds: an entry parameter's, an output's or
  temporaries; None for a constant's or a thread-local one."""
  if allocation.is_constant or allocation.is_thread_local:
    return None
  if allocation.is_entry_computation_parameter:
    return "parameter"
  return "output" if allocation.maybe_live_out else "temporary"
