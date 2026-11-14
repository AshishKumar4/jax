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
with XLA's own schema (``hlo_reader``) and then removes.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from jax._src import hlo_reader


@dataclasses.dataclass(frozen=True)
class LiveBuffer:
  """A temporary buffer live at a compile's temporary allocation's peak.

  Attributes:
    bytes: its size.
    shape: its dimensions, as compiled (a device's, after partitioning).
    dtype: its element type's XLA name (``f32``, ``bf16``, ``tuple``, ...).
  """
  bytes: int
  shape: tuple[int, ...]
  dtype: str


def dump_options(directory: str) -> dict[str, Any]:
  """The compile options that dump each of a compile's optimized modules,
  with its buffer assignment, as one uncompressed binary ``HloProto`` into
  ``directory``: every module, none per pass, whatever the dump settings the
  process or the caller gives (dumping disabled, or only unoptimized
  snapshots, which dump no module), all of them outside the persistent
  cache's key."""
  return {"xla_dump_to": directory, "xla_dump_hlo_as_proto": True,
          "xla_dump_compress_protos": False, "xla_dump_hlo_module_re": "",
          "xla_dump_hlo_pass_re": "", "xla_dump_max_hlo_modules": -1,
          "xla_enable_dumping": True,
          "xla_dump_hlo_unoptimized_snapshots": False}


def read(directory: str) -> tuple[LiveBuffer, ...] | None:
  """The buffers live at the peak of each temporary allocation of the
  modules dumped under ``directory`` (``dump_options``; a module to a
  subfolder of its own where the process asks it); None where no module was
  dumped with its buffer assignment: a compile the persistent cache answered,
  which dumps none, or, on XLA:CPU, the restored module alone, as its
  deserialization does."""
  hlo_proto = hlo_reader.message_class("xla.HloProto")
  primitive_type = hlo_reader.pool().FindEnumTypeByName("xla.PrimitiveType")
  assigned = []
  for folder, _, names in sorted(os.walk(directory)):
    for name in sorted(names):
      if name.endswith("after_optimizations.hlo.pb"):
        with open(os.path.join(folder, name), "rb") as f:
          hlo = hlo_proto.FromString(f.read())
        if hlo.HasField("buffer_assignment"):
          assigned.append(hlo)
  if not assigned:
    return None
  live: list[LiveBuffer] = []
  for hlo in assigned:
    shapes = {instruction.id: instruction.shape
              for computation in hlo.hlo_module.computations
              for instruction in computation.instructions}
    assignment = hlo.buffer_assignment
    buffers = {buffer.id: buffer for buffer in assignment.logical_buffers}
    # XLA's preallocated temporary buffer: no entry parameter's, constant's,
    # thread-local or output's allocation, whose buffers are all temporaries.
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
        shape = shapes[buffer.defined_at.instruction_id]
        for index in buffer.defined_at.shape_index:
          shape = shape.tuple_shapes[index]
        live.append(LiveBuffer(
            bytes=buffer.size, shape=tuple(shape.dimensions),
            dtype=primitive_type.values_by_number[shape.element_type]
            .name.lower()))
  return tuple(live)
