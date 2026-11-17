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
"""A compiled HLO module's instructions read through XLA's own schema (hlo.proto at the pinned XLA,
jax._src.xla_protos) from the module's serialization: each instruction's opcode, its custom-call target and the
computations it calls, by name. The module is decoded once; protobuf, which jax does not otherwise need, is
imported on first use."""

from __future__ import annotations

from collections.abc import Iterator
import functools
from typing import Any, NamedTuple


class Instruction(NamedTuple):
  """An instruction of a module (`instructions`): the computation holding it, its name and opcode, its
  custom-call target (None where it is no custom call), its operands and the computations it calls (a
  fusion's, a while loop's body and condition, a conditional's branches, a call's), each by name, whether
  it is its computation's root and whether its computation is the module's entry, and the instruction as
  XLA's schema holds it (`proto`, an `xla.HloInstructionProto`: its shape, dimension numbers and backend
  config, which the fields above do not name)."""

  computation: str
  name: str
  opcode: str
  custom_call_target: str | None
  operands: tuple[str, ...]
  called: tuple[str, ...]
  root: bool
  entry: bool
  proto: Any


@functools.cache
def _module_class() -> Any:
  try:
    from google.protobuf import descriptor_pb2
    from google.protobuf import descriptor_pool
    from google.protobuf import message_factory
  except ImportError as error:
    raise ImportError("reading an HLO module's instructions needs the protobuf package") from error
  from jax._src import xla_protos

  # A pool of its own: another library's copy of these files (xla/xla_data.proto) in the default pool
  # does not meet it.
  pool = descriptor_pool.DescriptorPool()
  for file in descriptor_pb2.FileDescriptorSet.FromString(xla_protos.FILE_DESCRIPTOR_SET).file:
    pool.Add(file)
  return message_factory.GetMessageClass(pool.FindMessageTypeByName("xla.HloModuleProto"))


def module_from_text(text: str) -> Any:
  """The HLO module `text` writes (an XLA dump file, a test's module), parsed by XLA's own HLO parser, which
  `instructions` reads as it reads a compiled one."""
  from jax._src.lib import hlo
  return hlo.hlo_module_from_text(text)


def instructions(module: Any) -> Iterator[Instruction]:
  """Each instruction of `module`, computation by computation in the module's order: an HLO module (a
  compiled executable's `hlo_modules()`, a lowering's `compiler_ir("hlo")`) or its serialized
  HloModuleProto."""
  if isinstance(module, (bytes, bytearray, memoryview)):
    serialized = bytes(module)
  else:
    serialized = module.as_serialized_hlo_module_proto()
  proto = _module_class().FromString(serialized)
  names = {computation.id: computation.name for computation in proto.computations}
  for computation in proto.computations:
    operands = {instruction.id: instruction.name for instruction in computation.instructions}
    entry = computation.id == proto.entry_computation_id
    for instruction in computation.instructions:
      yield Instruction(
          computation.name, instruction.name, instruction.opcode,
          instruction.custom_call_target if instruction.opcode == "custom-call" else None,
          tuple(operands[operand] for operand in instruction.operand_ids),
          tuple(names[called] for called in instruction.called_computation_ids),
          instruction.id == computation.root_id, entry, instruction)
