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
"""The buffers live at a compile's temporary allocation's peak.

XLA's buffer assignment names them (``BufferAssignmentProto.peak_buffers``),
but a GPU compile through a PJRT plugin returns only the assignment's sizes
(the C API's ``CompiledMemoryStats`` carries no assignment). A compile asked
for them (``Lowered.compile``'s ``peak_live_in``) has XLA dump its optimized
module with its buffer assignment as a binary ``HloProto``, which this reads
with XLA's own schema (``hlo_schema``) and then removes.
"""

from __future__ import annotations

import base64
import dataclasses
import functools
import os
from typing import Any, Literal

from jax._src import hlo_schema


@dataclasses.dataclass(frozen=True)
class LiveBuffer:
  """A buffer live at a compile's temporary allocation's peak.

  Attributes:
    bytes: its size.
    shape: its dimensions, as compiled (a device's, after partitioning).
    dtype: its element type's XLA name (``f32``, ``bf16``, ``tuple``, ...).
    kind: whether the entry computation's parameter, a constant or a
      temporary defines it.
    parameter: the entry parameter's number, for a parameter's.
  """
  bytes: int
  shape: tuple[int, ...]
  dtype: str
  kind: Literal["parameter", "constant", "temporary"]
  parameter: int | None


def dump_options(directory: str) -> dict[str, Any]:
  """The compile options that dump a compile's optimized module, with its
  buffer assignment, as a binary ``HloProto`` into ``directory``."""
  return {"xla_dump_to": directory, "xla_dump_hlo_as_proto": True}


@functools.cache
def _schema():
  from google.protobuf import descriptor_pb2  # pytype: disable=import-error
  from google.protobuf import descriptor_pool  # pytype: disable=import-error
  from google.protobuf import message_factory  # pytype: disable=import-error

  files = descriptor_pb2.FileDescriptorSet.FromString(
      base64.b64decode(hlo_schema.SERIALIZED))
  pool = descriptor_pool.DescriptorPool()
  for file in files.file:
    pool.Add(file)
  return (message_factory.GetMessageClass(pool.FindMessageTypeByName(
              "xla.HloProto")),
          pool.FindEnumTypeByName("xla.PrimitiveType"))


def read(directory: str) -> tuple[LiveBuffer, ...] | None:
  """The buffers live at the peak of each temporary allocation of the
  modules dumped into ``directory`` (``dump_options``); None where none was
  dumped (a compile the persistent cache answered)."""
  dumped = sorted(name for name in os.listdir(directory)
                  if name.endswith("after_optimizations.hlo.pb"))
  if not dumped:
    return None
  hlo_proto, primitive_type = _schema()
  live: list[LiveBuffer] = []
  for name in dumped:
    with open(os.path.join(directory, name), "rb") as f:
      hlo = hlo_proto.FromString(f.read())
    module, assignment = hlo.hlo_module, hlo.buffer_assignment
    instructions = {
        instruction.id: (instruction,
                         computation.id == module.entry_computation_id)
        for computation in module.computations
        for instruction in computation.instructions}
    buffers = {buffer.id: buffer for buffer in assignment.logical_buffers}
    temporary = {
        allocation.index for allocation in assignment.buffer_allocations
        if not (allocation.is_entry_computation_parameter
                or allocation.is_constant or allocation.is_thread_local
                or allocation.maybe_live_out)}
    for peak in assignment.peak_buffers:
      if peak.index not in temporary:
        continue
      for buffer_id in peak.logical_buffer_ids:
        buffer = buffers[buffer_id]
        instruction, in_entry = instructions[buffer.defined_at.instruction_id]
        shape = instruction.shape
        for index in buffer.defined_at.shape_index:
          shape = shape.tuple_shapes[index]
        kind = ("parameter" if instruction.opcode == "parameter" and in_entry
                else "constant" if instruction.opcode == "constant"
                else "temporary")
        live.append(LiveBuffer(
            bytes=buffer.size, shape=tuple(shape.dimensions),
            dtype=primitive_type.values_by_number[shape.element_type]
            .name.lower(),
            kind=kind,
            parameter=(instruction.parameter_number if kind == "parameter"
                       else None)))
  return tuple(live)
